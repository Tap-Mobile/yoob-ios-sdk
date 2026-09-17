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

@MainActor
final class ConversationTests: XCTestCase {
    private func event(_ object: [String: Any]) -> String { YoobConversation.encode(object) }

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
