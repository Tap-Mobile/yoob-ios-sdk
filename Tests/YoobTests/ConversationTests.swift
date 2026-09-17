import XCTest
@testable import Yoob

final class FakeSocket: RealtimeSocket, @unchecked Sendable {
    private let lock = NSLock()
    private var _sent: [[String: Any]] = []
    var sent: [[String: Any]] { lock.withLock { _sent } }
    func resume() {}
    func send(_ text: String) {
        let event = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:]
        lock.withLock { _sent.append(event) }
    }
    func receive() async throws -> String { try await Task.sleep(for: .seconds(3600)); return "" }
    func close() {}
    func clear() { lock.withLock { _sent = [] } }
}

/// A Yoob voice relay socket: records what was sent and can be closed by the "server" with a close code.
final class RelayFakeSocket: RealtimeSocket, @unchecked Sendable {
    private let lock = NSLock()
    private var _sent: [[String: Any]] = []
    private var _closeCode: Int?
    private let inbound: AsyncStream<String>
    private let feed: AsyncStream<String>.Continuation
    var sent: [[String: Any]] { lock.withLock { _sent } }
    var closeCode: Int? { lock.withLock { _closeCode } }

    init() { (inbound, feed) = AsyncStream.makeStream() }
    func resume() {}
    func send(_ text: String) {
        let event = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:]
        lock.withLock { _sent.append(event) }
    }
    func receive() async throws -> String {
        for await frame in inbound { return frame }
        throw URLError(.networkConnectionLost)
    }
    func close() { feed.finish() }
    func serverClose(_ code: Int) { lock.withLock { _closeCode = code }; feed.finish() }
}

/// Remembers the request the conversation connected with.
final class RequestBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _request: URLRequest?
    var request: URLRequest? { lock.withLock { _request } }
    func set(_ request: URLRequest) { lock.withLock { _request = request } }
}

@MainActor
final class ConversationTests: XCTestCase {
    private func event(_ object: [String: Any]) -> String { YoobConversation.encode(object) }

    nonisolated private static let relay = URL(string: "wss://voice.yoob.com/v1/realtime?model=gpt-realtime-2.1-mini")!
    nonisolated private static let session = YoobVoiceSession(voiceToken: "yv1.grant", url: relay, id: "vs_1")

    private func yoobConversation(_ options: YoobConversation.Options = .init(),
                                  session: YoobVoiceSession = ConversationTests.session)
        -> (YoobConversation, RelayFakeSocket, RequestBox) {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let socket = RelayFakeSocket()
        let box = RequestBox()
        let conversation = YoobConversation(avatar: avatar, options: options, voiceSession: { session }) { request in
            box.set(request)
            return socket
        }
        return (conversation, socket, box)
    }

    private func waitUntil(_ condition: () -> Bool) async {
        for _ in 0..<200 where !condition() { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testYoobVoiceConnectsWithTheGrantAndSendsOnlyVoiceAndInstructions() async throws {
        var options = YoobConversation.Options()
        options.voice = "marin"
        options.instructions = "You are Luna."
        // Ignored with Yoob voice: the relay sets these.
        options.model = "gpt-realtime"
        options.speed = 1.4
        options.turnDetection = .semantic()
        options.noiseReduction = "near_field"
        let (conversation, socket, box) = yoobConversation(options)
        _ = try await conversation.open()
        let request = try XCTUnwrap(box.request)
        XCTAssertEqual(request.url, Self.relay)
        XCTAssertEqual(request.value(forHTTPHeaderField: "Authorization"), "Bearer yv1.grant")
        XCTAssertEqual(request.value(forHTTPHeaderField: "Sec-WebSocket-Protocol"), "realtime")
        XCTAssertEqual(socket.sent.count, 2)
        XCTAssertEqual(socket.sent.compactMap { $0["type"] as? String }, ["session.update", "session.update"])
        let first = try XCTUnwrap(socket.sent[0]["session"] as? [String: Any])
        XCTAssertEqual(Array(first.keys), ["instructions"])
        XCTAssertEqual(first["instructions"] as? String, "You are Luna.")
        let second = try XCTUnwrap(socket.sent[1]["session"] as? [String: Any])
        XCTAssertEqual(Array(second.keys), ["audio"])
        let audio = try XCTUnwrap(second["audio"] as? [String: Any])
        XCTAssertEqual(Array(audio.keys), ["output"])
        XCTAssertEqual(audio["output"] as? [String: String], ["voice": "marin"])
    }

    func testYoobVoiceSendsNothingWhenTheBackendSetsVoiceAndInstructions() async throws {
        let (conversation, socket, _) = yoobConversation()
        _ = try await conversation.open()
        XCTAssertTrue(socket.sent.isEmpty)
    }

    func testOpenAIStillSendsTheFullSession() async throws {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let socket = RelayFakeSocket()
        let box = RequestBox()
        let conversation = YoobConversation(avatar: avatar, options: .init(), clientSecret: { "ek_test" }) {
            box.set($0)
            return socket
        }
        _ = try await conversation.open()
        XCTAssertEqual(box.request?.url?.absoluteString, "wss://api.openai.com/v1/realtime?model=gpt-realtime")
        XCTAssertEqual(box.request?.value(forHTTPHeaderField: "Authorization"), "Bearer ek_test")
        let session = try XCTUnwrap(socket.sent.first?["session"] as? [String: Any])
        let output = try XCTUnwrap((session["audio"] as? [String: Any])?["output"] as? [String: Any])
        XCTAssertEqual(output["speed"] as? Double, 1.08)
        // An OpenAI disconnect keeps its generic error.
        socket.serverClose(4002)
        await waitUntil { conversation.state == .ended }
        XCTAssertEqual(conversation.lastError, .network("the conversation disconnected"))
    }

    func testRelayCloseEndsTheConversationWithAClearError() async throws {
        let (conversation, socket, _) = yoobConversation()
        _ = try await conversation.open()
        socket.serverClose(4009)
        await waitUntil { conversation.state == .ended }
        XCTAssertEqual(conversation.state, .ended)
        XCTAssertEqual(conversation.lastError, .voiceSession(code: 4009, message: "This conversation reached its time limit."))
        XCTAssertEqual(conversation.lastError?.localizedDescription, "This conversation reached its time limit.")
    }

    func testEveryRelayCloseCodeHasAMessage() {
        let expected: [Int: String] = [
            1011: "The voice service disconnected. Start the conversation again.",
            1013: "Voice is busy right now. Try again in a moment.",
            4000: "The voice service refused this app's request. Update the app and try again.",
            4001: "The voice session was refused. Start the conversation again.",
            4002: "The voice session expired before it connected. Start the conversation again.",
            4003: "This voice session was already used. Start the conversation again.",
            4008: "This conversation reached its usage limit.",
            4009: "This conversation reached its time limit.",
            4010: "The conversation ended because it was idle for too long.",
            4029: "Voice has reached its usage limit for now. Try again later.",
            1006: "The conversation disconnected (1006).",
        ]
        for (code, message) in expected {
            XCTAssertEqual(YoobError.voiceClosed(code: code), .voiceSession(code: code, message: message))
        }
    }

    func testVoiceSessionDecodingMapsYoobAPIErrors() throws {
        let body = #"{"voice_session_id":"vs_1","voice_token":"yv1.t","url":"wss://voice.yoob.com/v1/realtime?model=m","model":"m","max_seconds":1800,"credits_per_minute":1.5,"expires_at":"2026-09-17T12:05:00Z"}"#
        let session = try JSONDecoder().decode(YoobVoiceSession.self, from: Data(body.utf8))
        XCTAssertEqual(session, YoobVoiceSession(voiceToken: "yv1.t", url: URL(string: "wss://voice.yoob.com/v1/realtime?model=m")!,
                                                 id: "vs_1", model: "m", maxSeconds: 1800, creditsPerMinute: 1.5,
                                                 expiresAt: "2026-09-17T12:05:00Z"))
        let cases: [(String, YoobError)] = [
            (#"{"code":"quota_exceeded"}"#, .outOfCredit),
            (#"{"error":"Invalid or revoked API key"}"#, .unauthorized),
            ("{}", .voiceSession(code: 0, message: "The backend didn't return a Yoob voice session.")),
        ]
        for (json, error) in cases {
            XCTAssertThrowsError(try JSONDecoder().decode(YoobVoiceSession.self, from: Data(json.utf8))) {
                XCTAssertEqual($0 as? YoobError, error)
            }
        }
    }

    func testYoobVoiceRefusesAnInsecureURL() async {
        let insecure = YoobVoiceSession(voiceToken: "t", url: URL(string: "http://voice.yoob.com/v1/realtime")!)
        let (conversation, _, box) = yoobConversation(session: insecure)
        do {
            _ = try await conversation.open()
            XCTFail("expected an error")
        } catch {
            XCTAssertEqual((error as? YoobError).map { if case .voiceSession = $0 { true } else { false } }, true)
        }
        XCTAssertNil(box.request)
    }

    func testYoobVoiceKeepsBargeInAndIgnoresLockedFields() {
        let (conversation, _, _) = yoobConversation()
        let socket = FakeSocket()
        conversation.attach(socket)
        conversation.handle(event(["type": "error", "error": ["code": "yoob_voice_locked", "message": "locked"]]))
        conversation.handle(event(["type": "error", "error": ["code": "yoob_instructions_locked"]]))
        XCTAssertNil(conversation.lastError)
        conversation.handle(event(["type": "response.created", "response": ["id": "r1"]]))
        conversation.handle(event(["type": "response.output_item.added", "response_id": "r1", "item": ["id": "i1"]]))
        conversation.handle(event(["type": "input_audio_buffer.speech_started"]))
        XCTAssertEqual(socket.sent.compactMap { $0["type"] as? String }, ["response.cancel", "conversation.item.truncate"])
        XCTAssertEqual(socket.sent.last?["item_id"] as? String, "i1")
        XCTAssertEqual(socket.sent.last?["audio_end_ms"] as? Int, 0)
        conversation.handle(event(["type": "error", "error": ["code": "yoob_invalid_audio", "message": "rejected"]]))
        XCTAssertEqual(conversation.lastError, .network("rejected"))
    }

    func testBothInitializersWorkWithTrailingClosures() {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let openAI = YoobConversation(avatar: avatar) { "ek" }
        let yoob = YoobConversation(avatar: avatar) { ConversationTests.session }
        XCTAssertEqual(openAI.state, .idle)
        XCTAssertEqual(yoob.state, .idle)
    }

    func testCancelledRepliesAreIgnoredAndBargeInTruncates() {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let conversation = YoobConversation(avatar: avatar, options: .init(), clientSecret: { "ek" }) { _ in FakeSocket() }
        let socket = FakeSocket()
        conversation.attach(socket)
        conversation.handle(event(["type": "response.created", "response": ["id": "r1"]]))
        conversation.handle(event(["type": "response.output_item.added", "response_id": "r1", "item": ["id": "i1"]]))
        conversation.handle(event(["type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": "Hi"]))
        XCTAssertEqual(conversation.assistantTranscript, "Hi")
        conversation.handle(event(["type": "input_audio_buffer.speech_started"]))
        XCTAssertEqual(socket.sent.compactMap { $0["type"] as? String }, ["response.cancel", "conversation.item.truncate"])
        XCTAssertEqual(socket.sent.last?["item_id"] as? String, "i1")
        XCTAssertEqual(conversation.state, .listening)
        // Late events from the cancelled reply change nothing.
        conversation.handle(event(["type": "response.output_audio_transcript.delta", "response_id": "r1", "delta": "stale"]))
        XCTAssertEqual(conversation.assistantTranscript, "Hi")
        conversation.handle(event(["type": "input_audio_buffer.speech_stopped"]))
        XCTAssertEqual(conversation.state, .thinking)
        conversation.handle(event(["type": "conversation.item.input_audio_transcription.completed", "transcript": "Hello"]))
        XCTAssertEqual(conversation.userTranscript, "Hello")
        conversation.handle(event(["type": "error", "error": ["code": "response_cancel_not_active"]]))
        XCTAssertNil(conversation.lastError)
        conversation.handle(event(["type": "error", "error": ["code": "rate_limit", "message": "Slow down"]]))
        XCTAssertEqual(conversation.lastError, .network("Slow down"))
    }

    func testTypedTextStartsAReply() {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let conversation = YoobConversation(avatar: avatar, options: .init(), clientSecret: { "ek" }) { _ in FakeSocket() }
        let socket = FakeSocket()
        conversation.attach(socket)
        conversation.send(text: "What's the weather?")
        XCTAssertEqual(socket.sent.compactMap { $0["type"] as? String }, ["conversation.item.create", "response.create"])
        XCTAssertEqual(conversation.state, .thinking)
    }
}
