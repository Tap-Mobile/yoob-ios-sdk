import XCTest
import LiveKit
import Yoob
@testable import YoobLiveKit

@MainActor
final class SessionTests: XCTestCase {
    func testCaptionsMergeStreamsBySegment() {
        var current: (id: String, stream: String)?
        var caption = ""
        // An agent reply: chunks of one stream.
        caption = YoobLiveKitSession.merge("Hello", into: caption, segment: "s1", stream: "a", current: &current)
        caption = YoobLiveKitSession.merge(" world", into: caption, segment: "s1", stream: "a", current: &current)
        XCTAssertEqual(caption, "Hello world")
        // A user turn: each stream repeats the full text so far.
        caption = YoobLiveKitSession.merge("Hello world!", into: caption, segment: "s1", stream: "b", current: &current)
        XCTAssertEqual(caption, "Hello world!")
        // A new segment is a new caption.
        caption = YoobLiveKitSession.merge("Next", into: caption, segment: "s2", stream: "c", current: &current)
        XCTAssertEqual(caption, "Next")
    }

    func testSessionRoutesCaptionsAndStops() async {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let session = YoobLiveKitSession(avatar: avatar, room: Room())
        XCTAssertEqual(session.state, .idle)
        session.receiveTranscript("Hi", stream: "a", segment: "1", from: "agent-1")
        session.receiveTranscript(" there", stream: "a", segment: "1", from: "agent-1")
        XCTAssertEqual(session.assistantTranscript, "Hi there")
        XCTAssertEqual(session.userTranscript, "")
        // Without an audio track the session still ends cleanly.
        await session.stop()
        XCTAssertEqual(session.state, .ended)
    }

    func testStateFollowsTheAgent() {
        typealias S = YoobLiveKitSession
        XCTAssertEqual(S.state(hasAgent: false, agentState: "speaking", isSpeaking: true), .connecting)
        XCTAssertEqual(S.state(hasAgent: true, agentState: "initializing", isSpeaking: false), .connecting)
        XCTAssertEqual(S.state(hasAgent: true, agentState: "listening", isSpeaking: true), .listening)
        XCTAssertEqual(S.state(hasAgent: true, agentState: "thinking", isSpeaking: false), .thinking)
        XCTAssertEqual(S.state(hasAgent: true, agentState: "speaking", isSpeaking: false), .speaking)
        XCTAssertEqual(S.state(hasAgent: true, agentState: "idle", isSpeaking: false), .listening)
        // No attribute: the silence gate decides.
        XCTAssertEqual(S.state(hasAgent: true, agentState: nil, isSpeaking: true), .speaking)
        XCTAssertEqual(S.state(hasAgent: true, agentState: nil, isSpeaking: false), .listening)
    }

    func testDrivesARealAvatarWithoutPlayingAudio() {
        // The avatar is not prepared (no renderer): it still accepts external-clock audio, ends and interrupts.
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent")))
        let segmenter = AgentSpeechSegmenter(sink: avatar)
        let start = ContinuousClock.now
        let voiced = { (ms: Int) in AgentAudioChunk(pcm: Data(repeating: 0x40, count: 480), level: 0.1, arrival: start + .milliseconds(ms)) }
        segmenter.setAgentState("speaking", at: start)
        for ms in stride(from: 0, to: 200, by: 10) { segmenter.receive(voiced(ms)) }
        XCTAssertEqual(avatar.stats.utterances, 1)
        segmenter.setAgentState("listening", at: start + .milliseconds(200))
        segmenter.tick(now: start + .seconds(1))
        XCTAssertFalse(segmenter.isActive)
        // A second reply starts a second utterance.
        segmenter.setAgentState("speaking", at: start + .seconds(1))
        segmenter.receive(voiced(1_000))
        XCTAssertEqual(avatar.stats.utterances, 2)
        segmenter.reset()
        XCTAssertEqual(avatar.phase, .notPrepared)
    }

    func testMeasuredLatencyIsSane() {
        let latency = YoobLiveKitSession.measuredOutputLatency()
        XCTAssertGreaterThanOrEqual(latency, .zero)
        XCTAssertLessThan(latency, .seconds(1))
    }
}
