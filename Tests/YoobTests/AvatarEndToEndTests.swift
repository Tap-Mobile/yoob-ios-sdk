import XCTest
@testable import Yoob

/// Runs the public API against packs on disk. Set YOOB_LOCAL_PACKS (a directory holding luna-realistic and luna-anime)
/// and YOOB_SPEECH_PCM (24 kHz mono PCM16).
@MainActor
final class AvatarEndToEndTests: XCTestCase {
    func testRealisticSpeaksWithExternalClock() async throws { try await speak("luna-realistic") }
    func testAnimeSpeaksWithExternalClock() async throws { try await speak("luna-anime") }

    private func speak(_ character: String) async throws {
        let env = ProcessInfo.processInfo.environment
        guard let packs = env["YOOB_LOCAL_PACKS"], let speech = env["YOOB_SPEECH_PCM"] else {
            throw XCTSkip("set YOOB_LOCAL_PACKS and YOOB_SPEECH_PCM")
        }
        let pcm = try Data(contentsOf: URL(fileURLWithPath: speech))
        setenv("YOOB_ALLOW_UNSIGNED_PACKS", "1", 1)  // development packs carry a plain character.json
        // A sandbox session token meters the run; a placeholder works too, since the test ends before the first beat.
        let session = YoobCredentials(sessionToken: env["YOOB_SESSION_TOKEN"] ?? "local-test", downloadToken: "unused")
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: packs).appendingPathComponent(character), credentials: { session }))
        try await avatar.prepare()
        XCTAssertEqual(avatar.phase, .ready)
        XCTAssertNotNil(avatar.poster)
        XCTAssertEqual(avatar.aspectRatio, 1080.0 / 1920.0, accuracy: 0.001)

        // Stream in 200 ms packets, as a realtime voice would, then report playback at real time.
        for offset in stride(from: 0, to: pcm.count, by: 9600) {
            try avatar.appendAudio(pcm: pcm.subdata(in: offset..<min(pcm.count, offset + 9600)))
        }
        avatar.endSpeech()
        XCTAssertEqual(avatar.phase, .speaking)
        let total = pcm.count / 2
        let clock = ContinuousClock(), start = clock.now
        while true {
            let elapsed = start.duration(to: clock.now)
            let played = min(total, Int(Double(elapsed.components.seconds) * 24000 + Double(elapsed.components.attoseconds) / 1e18 * 24000))
            avatar.audioPlayed(samples: played)
            if played >= total { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let expected = total * 25 / 24000
        print("\(character): shown \(avatar.stats.framesShown) of \(expected), skipped \(avatar.stats.framesSkipped), error \(avatar.lastRendererError ?? "none")")
        XCTAssertGreaterThan(avatar.stats.framesShown, expected / 2, "most frames should show on time")
        XCTAssertEqual(avatar.phase, .ready)
        XCTAssertFalse(avatar.isShowingSpeech)
        await avatar.close()
    }
}
