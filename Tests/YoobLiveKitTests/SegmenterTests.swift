import XCTest
@testable import YoobLiveKit

@MainActor
final class FakeAvatar: AvatarAudioSink {
    enum Call: Equatable { case append(Int), played(Int), end, interrupt }
    var calls: [Call] = []
    var appended: Int { calls.reduce(0) { if case .append(let n) = $1 { $0 + n } else { $0 } } }
    var lastPlayed: Int? { calls.reversed().lazy.compactMap { if case .played(let n) = $0 { n } else { nil } }.first }
    var ends: [Call] { calls.filter { $0 == .end || $0 == .interrupt } }

    func appendAudio(pcm: Data, sampleRate: Int) throws {
        XCTAssertEqual(sampleRate, 24_000)
        calls.append(.append(pcm.count / 2))
    }
    func audioPlayed(samples: Int) { calls.append(.played(samples)) }
    func endSpeech() { calls.append(.end) }
    func interrupt() -> Int { calls.append(.interrupt); return 0 }
}

@MainActor
final class SegmenterTests: XCTestCase {
    private let t0 = ContinuousClock.now
    private var clock: Duration = .zero
    private var now: ContinuousClock.Instant { t0 + clock }

    /// A 10 ms chunk (240 samples) arriving now; the clock then moves on 10 ms, as WebRTC's playout does.
    private func chunk(voiced: Bool) -> AgentAudioChunk {
        defer { clock += .milliseconds(10) }
        return AgentAudioChunk(pcm: Data(count: 480), level: voiced ? 0.1 : 0, arrival: now)
    }

    private func feed(_ segmenter: AgentSpeechSegmenter, voiced: Bool, count: Int) {
        for _ in 0..<count { segmenter.receive(chunk(voiced: voiced)) }
    }

    func testAgentStateStartsWithPreRollAndEndsAfterTrailingSilence() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        segmenter.setAgentState("listening", at: now)
        feed(segmenter, voiced: false, count: 50)
        XCTAssertTrue(avatar.calls.isEmpty)
        // The voice arrives 50 ms before the attribute: it is kept, the silence before it is not.
        feed(segmenter, voiced: true, count: 5)
        XCTAssertTrue(avatar.calls.isEmpty)
        segmenter.setAgentState("speaking", at: now)
        XCTAssertTrue(segmenter.isActive)
        XCTAssertEqual(avatar.appended, 5 * 240)
        feed(segmenter, voiced: true, count: 20)
        feed(segmenter, voiced: false, count: 10)   // the TTS's own trailing silence is still sent while speaking
        XCTAssertEqual(avatar.appended, 35 * 240)
        segmenter.setAgentState("listening", at: now)
        XCTAssertTrue(segmenter.isActive, "audio still in flight is followed")
        segmenter.receive(chunk(voiced: false))
        XCTAssertFalse(segmenter.isActive)
        XCTAssertEqual(avatar.ends, [.end])
        XCTAssertEqual(avatar.appended, 35 * 240, "the silence after the attribute is not sent")
        // Heard samples keep being reported until everything has played, then stop.
        clock += .seconds(1)
        segmenter.tick(now: now)
        XCTAssertEqual(avatar.lastPlayed, 35 * 240)
        let count = avatar.calls.count
        clock += .seconds(1)
        segmenter.tick(now: now)
        XCTAssertEqual(avatar.calls.count, count)
    }

    func testLeavingSpeakingMidWordInterrupts() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        segmenter.setAgentState("speaking", at: now)
        feed(segmenter, voiced: false, count: 3)
        XCTAssertFalse(segmenter.isActive, "leading silence does not start an utterance")
        feed(segmenter, voiced: true, count: 30)
        segmenter.setAgentState("listening", at: now)
        feed(segmenter, voiced: true, count: 2)      // in flight
        segmenter.receive(chunk(voiced: false))
        XCTAssertEqual(avatar.ends, [.interrupt])
        XCTAssertEqual(avatar.appended, 32 * 240)
        XCTAssertFalse(segmenter.isActive)
    }

    func testHangoverEndsAnUtteranceWhoseAudioKeepsGoing() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        segmenter.setAgentState("speaking", at: now)
        feed(segmenter, voiced: true, count: 10)
        segmenter.setAgentState("thinking", at: now)
        feed(segmenter, voiced: true, count: 39)
        XCTAssertTrue(segmenter.isActive)
        clock += .milliseconds(20)
        segmenter.tick(now: now)
        XCTAssertFalse(segmenter.isActive)
        XCTAssertEqual(avatar.ends.count, 1)
        // Audio after the end is not sent until the agent speaks again.
        let appended = avatar.appended
        feed(segmenter, voiced: true, count: 5)
        XCTAssertEqual(avatar.appended, appended)
        segmenter.setAgentState("speaking", at: now)
        XCTAssertTrue(segmenter.isActive)
    }

    func testSpeakingAgainWhileDrainingContinuesTheUtterance() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        segmenter.setAgentState("speaking", at: now)
        feed(segmenter, voiced: true, count: 10)
        segmenter.setAgentState("listening", at: now)
        segmenter.setAgentState("speaking", at: now)
        feed(segmenter, voiced: false, count: 5)
        feed(segmenter, voiced: true, count: 5)
        XCTAssertTrue(segmenter.isActive)
        XCTAssertEqual(avatar.ends, [])
        XCTAssertEqual(avatar.appended, 20 * 240)
    }

    func testSilenceGateWithoutAgentState() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        feed(segmenter, voiced: false, count: 10)
        XCTAssertTrue(avatar.calls.isEmpty)
        feed(segmenter, voiced: true, count: 10)
        XCTAssertTrue(segmenter.isActive)
        // A pause shorter than the gate is kept once the voice resumes.
        feed(segmenter, voiced: false, count: 30)
        XCTAssertEqual(avatar.appended, 10 * 240)
        feed(segmenter, voiced: true, count: 10)
        XCTAssertEqual(avatar.appended, 50 * 240)
        // A long pause ends it, and its silence is never sent.
        feed(segmenter, voiced: false, count: 59)
        XCTAssertTrue(segmenter.isActive)
        feed(segmenter, voiced: false, count: 1)
        XCTAssertFalse(segmenter.isActive)
        XCTAssertEqual(avatar.ends, [.end])
        XCTAssertEqual(avatar.appended, 50 * 240)
    }

    func testSilenceGateEndsWhenAudioStopsArriving() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        feed(segmenter, voiced: true, count: 10)
        clock += .milliseconds(590)
        segmenter.tick(now: now)
        XCTAssertTrue(segmenter.isActive)
        clock += .milliseconds(10)
        segmenter.tick(now: now)
        XCTAssertFalse(segmenter.isActive)
        XCTAssertEqual(avatar.ends, [.end])
    }

    func testHeardSamplesFollowPlayoutLatency() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar, latency: { .milliseconds(100) })
        let start = now
        // WebRTC can pull two 10 ms buffers in one I/O cycle: the second plays after the first.
        segmenter.receive(AgentAudioChunk(pcm: Data(count: 480), level: 0.1, arrival: start))
        segmenter.receive(AgentAudioChunk(pcm: Data(count: 480), level: 0.1, arrival: start))
        XCTAssertNil(avatar.lastPlayed)
        segmenter.tick(now: start + .milliseconds(100))
        XCTAssertNil(avatar.lastPlayed)
        segmenter.tick(now: start + .milliseconds(105))
        XCTAssertEqual(avatar.lastPlayed, 120)
        segmenter.tick(now: start + .milliseconds(115))
        XCTAssertEqual(avatar.lastPlayed, 360)
        segmenter.tick(now: start + .milliseconds(130))
        XCTAssertEqual(avatar.lastPlayed, 480)
        // Reports only move forward.
        let count = avatar.calls.count
        segmenter.tick(now: start + .milliseconds(110))
        XCTAssertEqual(avatar.calls.count, count)
    }

    func testResetInterruptsAndForgetsTheAgent() {
        let avatar = FakeAvatar()
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        var changes: [Bool] = []
        segmenter.onActiveChanged = { changes.append($0) }
        segmenter.setAgentState("speaking", at: now)
        feed(segmenter, voiced: true, count: 3)
        segmenter.reset()
        XCTAssertEqual(avatar.ends, [.interrupt])
        XCTAssertNil(segmenter.agentState)
        XCTAssertFalse(segmenter.isActive)
        XCTAssertEqual(changes, [true, false])
    }
}
