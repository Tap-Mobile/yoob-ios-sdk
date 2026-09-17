import XCTest
import CryptoKit
@testable import Yoob

/// Answers heartbeats from a script, records what was sent, and keeps a clock that moves only when the heartbeat sleeps.
final class HeartbeatScript: @unchecked Sendable {
    private let lock = NSLock()
    private var outcomes: [SessionAPI.Outcome]
    private let answer: ((Duration) -> SessionAPI.Outcome)?
    private var _tokens: [String] = []
    private var _sleeps: [Duration] = []
    private var _beatTimes: [Duration] = []
    private var _elapsed: Duration = .zero
    let start = ContinuousClock.now
    init(_ outcomes: [SessionAPI.Outcome]) { self.outcomes = outcomes; answer = nil }
    /// Answers by the time elapsed on the fake clock.
    init(answer: @escaping (Duration) -> SessionAPI.Outcome) { outcomes = []; self.answer = answer }
    var tokens: [String] { lock.withLock { _tokens } }
    var sleeps: [Duration] { lock.withLock { _sleeps } }
    var beatTimes: [Duration] { lock.withLock { _beatTimes } }
    var elapsed: Duration { lock.withLock { _elapsed } }
    var now: ContinuousClock.Instant { start + elapsed }
    func advance(_ duration: Duration) { lock.withLock { _elapsed += duration } }
    func next(_ credentials: YoobCredentials) -> SessionAPI.Outcome {
        lock.withLock {
            _tokens.append(credentials.sessionToken)
            _beatTimes.append(_elapsed)
            if let answer { return answer(_elapsed) }
            return outcomes.isEmpty ? .ok(.init()) : outcomes.removeFirst()
        }
    }
    func slept(_ duration: Duration) { lock.withLock { _sleeps.append(duration); _elapsed += duration } }
}

private extension Duration {
    static func minutes(_ value: Int) -> Duration { .seconds(value * 60) }
}

@MainActor
final class HeartbeatTests: XCTestCase {
    private var credentials: YoobCredentials? = YoobCredentials(sessionToken: "st_1", downloadToken: "yg1.first")
    private var ended: [YoobError] = []
    private var grants: [String] = []
    private var renewals = 0
    private var renewError: Error?
    private var events: [String] = []

    /// 2 s, 6 s, then every 15 s, like the default, without jitter.
    private static let backoff: (Int) -> Duration = { $0 <= 1 ? .seconds(2) : $0 == 2 ? .seconds(6) : .seconds(15) }

    /// Fails until `until`, alternating 503s and network errors, then answers.
    private static func outage(until: Duration) -> (Duration) -> SessionAPI.Outcome {
        var count = 0
        return { elapsed in
            count += 1
            if elapsed >= until { return .ok(.init()) }
            return count % 2 == 1 ? .transient("HTTP 503") : .transient("offline")
        }
    }

    private func heartbeat(_ script: HeartbeatScript, grace: Int = 600,
                           retryDelay: @escaping (Int) -> Duration = { .seconds($0) }) -> SessionHeartbeat {
        let heartbeat = SessionHeartbeat(
            hooks: .init(
                credentials: { [unowned self] in self.credentials },
                renew: { [unowned self] in
                    self.renewals += 1
                    if let renewError = self.renewError { throw renewError }
                    self.credentials = YoobCredentials(sessionToken: "st_\(self.renewals + 1)", downloadToken: "yg1.renewed")
                },
                grant: { [unowned self] token, _ in self.grants.append(token) },
                ended: { [unowned self] in self.ended.append($0) },
                degraded: { [unowned self] in self.events.append("degraded \($0)") },
                recovered: { [unowned self] in self.events.append("recovered") }),
            outageGraceSeconds: grace,
            retryDelay: retryDelay,
            send: { script.next($0) },
            sleep: { script.slept($0) },
            now: { script.now })
        heartbeat.start(automatic: false)
        return heartbeat
    }

    func testNineMinutesOfTransientFailuresKeepTheSessionThenItRecovers() async {
        let script = HeartbeatScript(answer: Self.outage(until: .minutes(9)))
        let beat = heartbeat(script, retryDelay: Self.backoff)
        await beat.beatNow()
        XCTAssertTrue(ended.isEmpty)
        XCTAssertTrue(beat.isRunning)
        XCTAssertFalse(beat.isDegraded)
        XCTAssertEqual(beat.consecutiveFailures, 0)
        XCTAssertGreaterThan(script.tokens.count, 30, "retried through the outage")
        XCTAssertEqual(Array(script.sleeps.prefix(4)), [.seconds(2), .seconds(6), .seconds(15), .seconds(15)])
        XCTAssertGreaterThanOrEqual(script.beatTimes.last!, .minutes(9))
        XCTAssertEqual(events, ["degraded HTTP 503", "recovered"])
    }

    func testTheGraceWindowRestartsAfterARecovery() async {
        let first = Self.outage(until: .minutes(9)), second = Self.outage(until: .minutes(19))
        let script = HeartbeatScript(answer: { $0 < .minutes(10) ? first($0) : second($0) })
        let beat = heartbeat(script, retryDelay: Self.backoff)
        await beat.beatNow()
        script.advance(.minutes(10) - script.elapsed)
        await beat.beatNow()
        XCTAssertTrue(ended.isEmpty, "two 9-minute outages with a success between them are both survived")
        XCTAssertEqual(events, ["degraded HTTP 503", "recovered", "degraded HTTP 503", "recovered"])
    }

    func testTenMinutesOfTransientFailuresEndTheSessionAsUnreachable() async {
        let script = HeartbeatScript(answer: { _ in .transient("HTTP 502") })
        let beat = heartbeat(script, retryDelay: Self.backoff)
        await beat.beatNow()
        XCTAssertEqual(ended, [.sessionEnded("unreachable")])
        XCTAssertEqual(ended.first?.errorDescription, "Yoob couldn't be reached, so the session ended.")
        XCTAssertFalse(beat.isRunning)
        let times = script.beatTimes
        XCTAssertGreaterThanOrEqual(times.last!, .minutes(10), "the last attempt lands on the deadline")
        XCTAssertLessThan(times.last!, .minutes(10) + .seconds(15))
        XCTAssertLessThan(times[times.count - 2], .minutes(10), "still retrying inside the window")
        XCTAssertEqual(events, ["degraded HTTP 502"])
        await beat.beatNow()
        XCTAssertEqual(script.tokens.count, times.count, "no beats after the session ended")
    }

    func testA402StopsAtOnceEvenWhileDegraded() async {
        let script = HeartbeatScript(answer: { $0 < .minutes(3) ? .transient("offline") : .fatal(.outOfCredit) })
        let beat = heartbeat(script, retryDelay: Self.backoff)
        await beat.beatNow()
        XCTAssertEqual(ended, [.outOfCredit])
        XCTAssertLessThan(script.beatTimes.last!, .minutes(4), "did not wait for the grace window")
        XCTAssertEqual(events, ["degraded offline"])
        XCTAssertFalse(beat.isRunning)

        ended = []; events = []
        let stopped = HeartbeatScript(answer: { $0 < .minutes(5) ? .transient("HTTP 500") : .ok(.init(stop: true, reason: "key-revoked")) })
        let revoked = heartbeat(stopped, retryDelay: Self.backoff)
        await revoked.beatNow()
        XCTAssertEqual(ended, [.unauthorized])
        XCTAssertFalse(revoked.isRunning)
    }

    func testTheGraceWindowIsConfigurable() async {
        let strict = HeartbeatScript(answer: { _ in .transient("offline") })
        await heartbeat(strict, grace: 0).beatNow()
        XCTAssertEqual(strict.tokens.count, 1, "0 ends at the first failure")
        XCTAssertEqual(ended, [.sessionEnded("unreachable")])

        ended = []
        let short = HeartbeatScript(answer: Self.outage(until: .seconds(90)))
        await heartbeat(short, grace: 60, retryDelay: Self.backoff).beatNow()
        XCTAssertEqual(ended, [.sessionEnded("unreachable")])
        XCTAssertGreaterThanOrEqual(short.beatTimes.last!, .seconds(60))
        XCTAssertLessThan(short.beatTimes.last!, .seconds(90))

        ended = []
        let capped = HeartbeatScript(answer: { _ in .transient("HTTP 500") })
        let long = heartbeat(capped, grace: 99_999, retryDelay: Self.backoff)
        XCTAssertEqual(long.outageGrace, .seconds(1800))
        await long.beatNow()
        XCTAssertGreaterThanOrEqual(capped.beatTimes.last!, .minutes(30))
        XCTAssertLessThan(capped.beatTimes.last!, .minutes(30) + .seconds(15))
        XCTAssertEqual(heartbeat(HeartbeatScript([]), grace: -5).outageGrace, .zero)

        let avatar = YoobAvatar(.cloud(character: "luna-anime") { throw YoobError.unauthorized })
        XCTAssertEqual(avatar.heartbeatOutageGraceSeconds, 600)
        avatar.heartbeatOutageGraceSeconds = 5000
        XCTAssertEqual(avatar.heartbeatOutageGraceSeconds, 1800)
        XCTAssertEqual(YoobAvatar(.cloud(character: "luna-anime") { throw YoobError.unauthorized }, heartbeatOutageGraceSeconds: -1)
            .heartbeatOutageGraceSeconds, 0)
    }

    func testTimeTheAppWasSuspendedDoesNotCountAgainstTheGraceWindow() async {
        let script = HeartbeatScript(answer: Self.outage(until: .minutes(40)))
        let beat = heartbeat(script, retryDelay: Self.backoff)
        script.advance(.minutes(30))
        await beat.beatNow()
        XCTAssertEqual(ended, [.sessionEnded("unreachable")])
        let times = script.beatTimes
        XCTAssertGreaterThanOrEqual(times.last! - times.first!, .minutes(10) - .seconds(30), "retried for 10 minutes less one 30 s interval after resuming")
    }

    func testRefusedOrExhaustedSessionsStopAtOnce() async {
        let cases: [(SessionAPI.Outcome, YoobError)] = [
            (.fatal(.unauthorized), .unauthorized),
            (.fatal(.outOfCredit), .outOfCredit),
            (.ok(.init(stop: true, reason: "out-of-credits")), .outOfCredit),
            (.ok(.init(stop: true, reason: "sandbox-limit")), .sessionEnded("this sandbox session reached its time limit")),
            (.ok(.init(stop: true, reason: "key-revoked")), .unauthorized),
        ]
        for (outcome, expected) in cases {
            ended = []
            let script = HeartbeatScript([outcome])
            let beat = heartbeat(script)
            await beat.beatNow()
            XCTAssertEqual(script.tokens.count, 1)
            XCTAssertEqual(ended, [expected])
            XCTAssertFalse(beat.isRunning)
        }
    }

    func testAForgottenSessionIsRenewedThroughTheBackend() async {
        let script = HeartbeatScript([.gone, .ok(.init()), .ok(.init(stop: true, reason: "abandoned"))])
        let beat = heartbeat(script)
        await beat.beatNow()
        XCTAssertEqual(renewals, 1)
        await beat.beatNow()
        XCTAssertEqual(script.tokens, ["st_1", "st_2"], "the renewed session beats")
        await beat.beatNow()
        XCTAssertEqual(renewals, 2)
        XCTAssertTrue(ended.isEmpty)

        renewError = URLError(.notConnectedToInternet)
        let failing = heartbeat(HeartbeatScript([.gone]))
        await failing.beatNow()
        guard case .sessionEnded = ended.first else { return XCTFail("expected sessionEnded, got \(ended)") }

        ended = []
        renewError = YoobError.outOfCredit
        let broke = heartbeat(HeartbeatScript([.ok(.init(stop: true, reason: "abandoned"))]))
        await broke.beatNow()
        XCTAssertEqual(ended, [.outOfCredit])
    }

    func testRenewedGrantsAreHandedOver() async {
        let script = HeartbeatScript([.ok(.init(grant: "yg1.renewed", grantExpiresAt: "1790000000")), .ok(.init(grant: ""))])
        let beat = heartbeat(script)
        await beat.beatNow()
        await beat.beatNow()
        XCTAssertEqual(grants, ["yg1.renewed"])
        XCTAssertTrue(ended.isEmpty)
    }

    func testNoCredentialsFailsClosed() async {
        credentials = nil
        let script = HeartbeatScript([])
        await heartbeat(script).beatNow()
        XCTAssertTrue(script.tokens.isEmpty)
        XCTAssertEqual(ended, [.unauthorized])
    }

    func testHeartbeatRepliesDecodeWithAndWithoutAGrant() {
        let cases: [(Int, String, SessionAPI.Outcome)] = [
            (200, #"{"stop":false,"credits_remaining":10,"billed_seconds":15,"reason":null}"#, .ok(.init())),
            (200, #"{"stop":true,"reason":"out-of-credits"}"#, .ok(.init(stop: true, reason: "out-of-credits"))),
            (200, #"{"stop":false,"grant":"yg1.x","grant_expires_at":"2026-09-17T13:00:00Z"}"#,
             .ok(.init(grant: "yg1.x", grantExpiresAt: "2026-09-17T13:00:00Z"))),
            (200, #"{"grant":"yg1.x","grant_expires_at":1790000000}"#, .ok(.init(grant: "yg1.x", grantExpiresAt: "1790000000"))),
            (200, #"{"grant":null}"#, .ok(.init())),
            (200, #"{"stop":false,"credits_remaining":9,"billed_seconds":15,"reason":null,"download_token":"yg1.d","download_token_expires_at":1790000000}"#,
             .ok(.init(grant: "yg1.d", grantExpiresAt: "1790000000"))),
            (200, #"{"download_token":"yg1.d","grant":"yg1.old"}"#, .ok(.init(grant: "yg1.d"))),
            (200, #"{"download_token":"","grant":"yg1.old","grant_expires_at":"soon"}"#, .ok(.init(grant: "yg1.old", grantExpiresAt: "soon"))),
            (402, #"{"stop":true,"reason":"suspended","code":"suspended"}"#, .fatal(.sessionEnded("this Yoob workspace is suspended"))),
            (403, #"{"stop":true,"reason":"key-revoked","code":"key_revoked"}"#, .fatal(.unauthorized)),
            (402, #"{"code":"quota_exceeded"}"#, .fatal(.outOfCredit)),
            (200, "", .ok(.init())),
            (200, "not json", .transient("unreadable heartbeat reply")),
            (401, "{}", .fatal(.unauthorized)),
            (403, "{}", .fatal(.unauthorized)),
            (402, "{}", .fatal(.outOfCredit)),
            (404, #"{"error":"Unknown or already-ended session."}"#, .gone),
            (409, "{}", .transient("HTTP 409")),
            (503, "", .transient("HTTP 503")),
        ]
        for (status, body, expected) in cases {
            XCTAssertEqual(SessionAPI.outcome(status: status, body: Data(body.utf8)), expected, "\(status) \(body)")
        }
    }

    func testCredentialsKeepTheSessionWhenTheGrantIsRenewed() {
        let renewed = credentials!.renewingGrant("yg1.second")
        XCTAssertEqual(renewed.sessionToken, "st_1")
        XCTAssertEqual(renewed.downloadToken, "yg1.second")
        XCTAssertEqual(renewed.heartbeatSeconds, credentials!.heartbeatSeconds)
        let access = CDNAccess(credentials!)
        access.downloadToken = "yg1.third"
        XCTAssertEqual(access.downloadToken, "yg1.third")
    }

    func testSessionEndedHasAReadableMessage() {
        XCTAssertEqual(YoobError.sessionEnded("x").localizedDescription, "The Yoob session ended: x.")
    }
}

@MainActor
final class VoiceHostTests: XCTestCase {
    func testOnlyYoobHostsByDefault() {
        let allowed = ["wss://voice.yoob.com/v1/realtime?model=m", "wss://VOICE.yoob.com./v1/realtime", "wss://eu.voice.yoob.com/"]
        for url in allowed {
            XCTAssertTrue(YoobConversation.isAllowedVoiceURL(URL(string: url)!, hosts: YoobConversation.defaultVoiceHosts), url)
        }
        let refused = ["wss://yoob.com/", "ws://voice.yoob.com/", "wss://voice.yoob.com.attacker.io/", "wss://evilyoob.com/",
                       "wss://user:pw@voice.yoob.com/", "https://voice.yoob.com/"]
        for url in refused {
            XCTAssertFalse(YoobConversation.isAllowedVoiceURL(URL(string: url)!, hosts: YoobConversation.defaultVoiceHosts), url)
        }
        XCTAssertTrue(YoobConversation.isAllowedVoiceURL(URL(string: "wss://relay.example.com/")!, hosts: ["relay.example.com"]))
        XCTAssertFalse(YoobConversation.isAllowedVoiceURL(URL(string: "wss://relay.example.com/")!, hosts: ["example.com"]))
        XCTAssertFalse(YoobConversation.isAllowedVoiceURL(URL(string: "wss://relay.example.com/")!, hosts: ["*."]))
    }

    func testConversationRefusesOtherHostsUnlessConfigured() async throws {
        let session = YoobVoiceSession(voiceToken: "t", url: URL(string: "wss://relay.example.com/v1/realtime")!)
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent"), credentials: { throw YoobError.unauthorized }))
        let box = RequestBox()
        let refused = YoobConversation(avatar: avatar, options: .init(), voiceSession: { session }) {
            box.set($0); return RelayFakeSocket()
        }
        do {
            _ = try await refused.open()
            XCTFail("expected an error")
        } catch let YoobError.voiceSession(_, message) {
            XCTAssertTrue(message.contains("allowed host"), message)
        }
        XCTAssertNil(box.request)

        var options = YoobConversation.Options()
        options.voiceHosts = ["relay.example.com"]
        let selfHosted = YoobConversation(avatar: avatar, options: options, voiceSession: { session }) {
            box.set($0); return RelayFakeSocket()
        }
        _ = try await selfHosted.open()
        XCTAssertEqual(box.request?.url, session.url)
    }
}

final class LocalPackTests: XCTestCase {
    private var directory: URL!
    private let signingKey = Curve25519.Signing.PrivateKey()
    private var keys: [String: Data] { ["test-key": signingKey.publicKey.rawRepresentation] }

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent("yoob-local-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        unsetenv("YOOB_ALLOW_UNSIGNED_PACKS")
    }

    override func tearDownWithError() throws {
        unsetenv("YOOB_ALLOW_UNSIGNED_PACKS")
        try? FileManager.default.removeItem(at: directory)
    }

    /// Writes a one-file pack and returns its manifest bytes.
    private func writePack(poster: Data = Data("poster".utf8)) throws -> Data {
        try poster.write(to: directory.appendingPathComponent("poster.jpg"))
        let sha = Hex.string(SHA256.hash(data: poster))
        let manifest: [String: Any] = [
            "schema": 1, "character": "luna-test", "version": "2026.09.17.1", "engine": "anime", "displayName": "Luna",
            "width": 1080, "height": 1920, "poster": "poster.jpg", "idle": ["frames": [], "fps": 8], "minSDK": "0.1.0",
            "files": [["path": "poster.jpg", "size": poster.count, "sha256": sha, "tier": 0,
                       "chunks": [["sha256": sha, "size": poster.count]]]],
        ]
        return try JSONSerialization.data(withJSONObject: manifest)
    }

    private func sign(_ payload: Data, keyId: String = "test-key") throws {
        let signature = try signingKey.signature(for: payload)
        let envelope = ["keyId": keyId, "payload": payload.base64EncodedString(), "signature": signature.base64EncodedString()]
        try JSONSerialization.data(withJSONObject: envelope).write(to: directory.appendingPathComponent(LocalPack.signedManifestName))
    }

    private func assertRejected(_ contains: String, file: StaticString = #filePath, line: UInt = #line) async {
        do {
            _ = try await LocalPack.open(directory, keys: keys)
            XCTFail("expected the pack to be refused", file: file, line: line)
        } catch let YoobError.invalidAssets(detail) {
            XCTAssertTrue(detail.contains(contains), "\(detail) should mention \(contains)", file: file, line: line)
        } catch {
            XCTFail("unexpected \(error)", file: file, line: line)
        }
    }

    func testASignedPackWithMatchingFilesOpens() async throws {
        try sign(try writePack())
        let manifest = try await LocalPack.open(directory, keys: keys)
        XCTAssertEqual(manifest.character, "luna-test")
    }

    func testUnsignedPacksAreRefused() async throws {
        let payload = try writePack()
        try payload.write(to: directory.appendingPathComponent(LocalPack.unsignedManifestName))
        await assertRejected("signed manifest")
    }

    func testUnsignedPacksOpenOnlyWithTheDebugOptIn() async throws {
        let payload = try writePack()
        try payload.write(to: directory.appendingPathComponent(LocalPack.unsignedManifestName))
        setenv("YOOB_ALLOW_UNSIGNED_PACKS", "1", 1)
        #if DEBUG
        let manifest = try await LocalPack.open(directory, keys: keys)
        XCTAssertEqual(manifest.character, "luna-test")
        #else
        await assertRejected("signed manifest")
        #endif
    }

    func testBadSignaturesAndUnknownKeysAreRefused() async throws {
        let payload = try writePack()
        try sign(payload, keyId: "someone-else")
        await assertRejected("unknown signing key")
        try sign(payload)
        var tampered = payload
        tampered[tampered.startIndex] = UInt8(ascii: " ")
        let envelope = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent(LocalPack.signedManifestName))) as! [String: String]
        let forged = ["keyId": "test-key", "payload": tampered.base64EncodedString(), "signature": envelope["signature"]!]
        try JSONSerialization.data(withJSONObject: forged).write(to: directory.appendingPathComponent(LocalPack.signedManifestName))
        await assertRejected("manifest signature")
    }

    func testModifiedOrMissingFilesAreRefused() async throws {
        try sign(try writePack())
        try Data("POSTER".utf8).write(to: directory.appendingPathComponent("poster.jpg"))
        await assertRejected("poster.jpg failed verification")
        try FileManager.default.removeItem(at: directory.appendingPathComponent("poster.jpg"))
        await assertRejected("poster.jpg")
    }

    func testTheProductionKeyRejectsATestSignature() async throws {
        try sign(try writePack())
        do {
            _ = try await LocalPack.open(directory)
            XCTFail("expected the pack to be refused")
        } catch let YoobError.invalidAssets(detail) {
            XCTAssertTrue(detail.contains("unknown signing key"), detail)
        }
    }

    @MainActor
    func testALocalAvatarNeedsASessionBeforeOpeningThePack() async throws {
        try sign(try writePack())
        let avatar = YoobAvatar(.local(directory, credentials: { throw YoobError.unauthorized }))
        do {
            try await avatar.prepare()
            XCTFail("expected prepare() to fail without a session")
        } catch {
            XCTAssertEqual(error as? YoobError, .unauthorized)
        }
        XCTAssertEqual(avatar.phase, .failed(.unauthorized))
        XCTAssertNil(avatar.manifest, "the pack isn't opened without a session")
    }
}
