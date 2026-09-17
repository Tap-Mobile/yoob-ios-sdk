import XCTest
@testable import Yoob

/// A Gemini Live socket: answers the setup the way the test asks and delivers queued server frames.
final class GeminiFakeSocket: RealtimeSocket, @unchecked Sendable {
    enum SetupReply { case complete, error(String), close(String) }
    private let lock = NSLock()
    private var _sent: [[String: Any]] = []
    private let inbound: AsyncStream<String>
    private let feed: AsyncStream<String>.Continuation
    private let setupReply: SetupReply
    private var _closeReason: String?
    var sent: [[String: Any]] { lock.withLock { _sent } }
    var closeReason: String? { lock.withLock { _closeReason } }

    init(setupReply: SetupReply = .complete) {
        self.setupReply = setupReply
        (inbound, feed) = AsyncStream.makeStream()
    }
    func resume() {}
    func send(_ text: String) {
        let message = (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:]
        lock.withLock { _sent.append(message) }
        guard message["setup"] != nil else { return }
        switch setupReply {
        case .complete: server(["setupComplete": [String: Any]()])
        case .error(let text): server(["error": ["code": 400, "message": text]])
        case .close(let reason): disconnect(reason)
        }
    }
    func receive() async throws -> String {
        for await frame in inbound { return frame }
        throw URLError(.networkConnectionLost)
    }
    func close() { feed.finish() }
    func server(_ message: [String: Any]) { feed.yield(YoobConversation.encode(message)) }
    func disconnect(_ reason: String) { lock.withLock { _closeReason = reason }; feed.finish() }
}

@MainActor
final class GeminiConversationTests: XCTestCase {
    private func makeConversation(_ options: YoobGeminiConversation.Options = .init(),
                                  socket: GeminiFakeSocket = GeminiFakeSocket()) -> YoobGeminiConversation {
        let avatar = YoobAvatar(.local(URL(fileURLWithPath: "/nonexistent"), credentials: { throw YoobError.unauthorized }))
        return YoobGeminiConversation(avatar: avatar, options: options, token: { "t" }) { _ in socket }
    }

    private func audio(_ samples: [Int16], rate: Int = 24000) -> [String: Any] {
        let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
        return ["serverContent": ["modelTurn": ["parts": [["inlineData": [
            "mimeType": "audio/pcm;rate=\(rate)", "data": data.base64EncodedString()]]]]]]
    }

    private func settle(_ timeout: TimeInterval = 1, until condition: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() && Date() < deadline { try? await Task.sleep(for: .milliseconds(10)) }
    }

    func testGeminiRequestUsesConstrainedEndpointAndEscapedToken() {
        let url = YoobGeminiConversation.request(token: "auth_tokens/a+b=").url!.absoluteString
        XCTAssertEqual(url, "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta"
                       + ".GenerativeService.BidiGenerateContentConstrained?access_token=auth_tokens%2Fa%2Bb%3D")
    }

    func testGeminiSetupIsSentFirstAndAwaited() async throws {
        var options = YoobGeminiConversation.Options()
        options.voice = "Kore"
        options.systemInstruction = "You are Luna."
        let socket = GeminiFakeSocket()
        let conversation = makeConversation(options, socket: socket)
        _ = try await conversation.open(token: "t")
        XCTAssertEqual(socket.sent.count, 1)
        let setup = try XCTUnwrap(socket.sent.first?["setup"] as? [String: Any])
        let expected: [String: Any] = [
            "model": "models/gemini-3.8-live",
            "generationConfig": ["responseModalities": ["AUDIO"],
                                 "speechConfig": ["voiceConfig": ["prebuiltVoiceConfig": ["voiceName": "Kore"]]]],
            "systemInstruction": ["parts": [["text": "You are Luna."]]],
            "realtimeInputConfig": [
                "automaticActivityDetection": [
                    "disabled": false, "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                    "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH", "prefixPaddingMs": 100, "silenceDurationMs": 450,
                ],
                "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            ],
            "inputAudioTranscription": [String: Any](),
            "outputAudioTranscription": [String: Any](),
        ]
        XCTAssertEqual(setup as NSDictionary, expected as NSDictionary)
        conversation.stop()
    }

    func testGeminiSetupOptions() {
        var options = YoobGeminiConversation.Options()
        options.model = "models/gemini-3.8-live-extended-thinking"
        options.activityDetection.startSensitivity = .low
        options.activityDetection.endSensitivity = .low
        options.activityDetection.silenceMS = 800
        options.inputTranscription = false
        options.outputTranscription = false
        let setup = makeConversation(options).setupMessage()
        XCTAssertEqual(setup["model"] as? String, "models/gemini-3.8-live-extended-thinking")
        let vad = (setup["realtimeInputConfig"] as? [String: Any])?["automaticActivityDetection"] as? [String: Any]
        XCTAssertEqual(vad?["startOfSpeechSensitivity"] as? String, "START_SENSITIVITY_LOW")
        XCTAssertEqual(vad?["endOfSpeechSensitivity"] as? String, "END_SENSITIVITY_LOW")
        XCTAssertEqual(vad?["silenceDurationMs"] as? Int, 800)
        XCTAssertNil(setup["inputAudioTranscription"])
        XCTAssertNil(setup["outputAudioTranscription"])
        XCTAssertNil(setup["systemInstruction"])
        XCTAssertNil((setup["generationConfig"] as? [String: Any])?["speechConfig"])
    }

    func testGeminiSetupFailuresThrow() async {
        do {
            _ = try await makeConversation(socket: GeminiFakeSocket(setupReply: .close("1008: token expired"))).open(token: "t")
            XCTFail("expected a failure")
        } catch {
            XCTAssertEqual(error as? YoobError, .network("Gemini Live closed the connection (1008: token expired)"))
        }
        do {
            _ = try await makeConversation(socket: GeminiFakeSocket(setupReply: .error("Invalid voice"))).open(token: "t")
            XCTFail("expected a failure")
        } catch {
            XCTAssertEqual(error as? YoobError, .network("Invalid voice"))
        }
    }

    func testGeminiMicrophoneAudioIsResampledTo16kHz() throws {
        let socket = GeminiFakeSocket()
        let uplink = try GeminiAudioUplink(socket: socket)
        var total = 0
        for packet in 0..<50 {
            let samples = (0..<480).map { Int16(8000 * sin(2 * Double.pi * 440 * Double(packet * 480 + $0) / 24000)) }
            uplink.send(samples.withUnsafeBufferPointer { Data(buffer: $0) })
        }
        for message in socket.sent {
            let audio = try XCTUnwrap((message["realtimeInput"] as? [String: Any])?["audio"] as? [String: Any])
            XCTAssertEqual(audio["mimeType"] as? String, "audio/pcm;rate=16000")
            total += try XCTUnwrap(Data(base64Encoded: audio["data"] as? String ?? "")).count / 2
        }
        XCTAssertGreaterThanOrEqual(socket.sent.count, 45)
        XCTAssertEqual(Double(total), 16000, accuracy: 400, "one second of audio")
    }

    func testGeminiResamplerKeepsSpeechAndRemovesAliases() throws {
        func rms(_ data: Data) -> Double {
            let samples = data.withUnsafeBytes { Array($0.bindMemory(to: Int16.self)) }.dropFirst(200)
            return sqrt(samples.reduce(0) { $0 + Double($1) * Double($1) } / Double(samples.count))
        }
        func tone(_ hz: Double) -> Data {
            let samples = (0..<24000).map { Int16(10000 * sin(2 * Double.pi * hz * Double($0) / 24000)) }
            return samples.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        let speech = try PCM16Resampler(from: 24000, to: 16000).convert(tone(440))
        XCTAssertEqual(rms(speech), 10000 / 2.squareRoot(), accuracy: 300)
        let alias = try PCM16Resampler(from: 24000, to: 16000).convert(tone(10000))
        XCTAssertLessThan(rms(alias), 700)
        XCTAssertThrowsError(try PCM16Resampler(from: 24000, to: 16000).convert(Data([1, 2, 3])))
        XCTAssertEqual(YoobGeminiConversation.sampleRate("audio/pcm;rate=16000"), 16000)
        XCTAssertEqual(YoobGeminiConversation.sampleRate("audio/pcm"), 24000)
    }

    func testGeminiRepliesTranscriptsAndTurnCompletion() async {
        let conversation = makeConversation()
        conversation.attach(GeminiFakeSocket())
        conversation.handle(YoobConversation.encode(["serverContent": ["inputTranscription": ["text": "Hola,"]]]))
        conversation.handle(YoobConversation.encode(["serverContent": ["inputTranscription": ["text": " qué tal"]]]))
        XCTAssertEqual(conversation.userTranscript, "Hola, qué tal")
        conversation.handle(YoobConversation.encode(audio(Array(repeating: 500, count: 2400))))
        XCTAssertEqual(conversation.state, .speaking)
        conversation.handle(YoobConversation.encode(["serverContent": ["outputTranscription": ["text": "Muy "]]]))
        conversation.handle(YoobConversation.encode(["serverContent": ["outputTranscription": ["text": "bien."]]]))
        XCTAssertEqual(conversation.assistantTranscript, "Muy bien.")
        conversation.handle(YoobConversation.encode(["serverContent": ["generationComplete": true]]))
        conversation.handle(YoobConversation.encode(["serverContent": ["turnComplete": true]]))
        await settle { conversation.state == .listening }
        XCTAssertEqual(conversation.state, .listening)
        XCTAssertEqual(conversation.assistantTranscript, "Muy bien.")
        // The next utterance starts a fresh caption; the next reply a fresh transcript.
        conversation.handle(YoobConversation.encode(["serverContent": ["inputTranscription": ["text": "Gracias"]]]))
        XCTAssertEqual(conversation.userTranscript, "Gracias")
        conversation.handle(YoobConversation.encode(["serverContent": ["outputTranscription": ["text": "De nada."]]]))
        XCTAssertEqual(conversation.assistantTranscript, "De nada.")
        XCTAssertNil(conversation.lastError)
    }

    func testGeminiInterruptionAndTypedText() {
        let socket = GeminiFakeSocket()
        let conversation = makeConversation(socket: socket)
        conversation.attach(socket)
        conversation.handle(YoobConversation.encode(audio([5, 6])))
        conversation.handle(YoobConversation.encode(["serverContent": ["outputTranscription": ["text": "Let me tell"]]]))
        XCTAssertEqual(conversation.state, .speaking)
        conversation.handle(YoobConversation.encode(["serverContent": ["interrupted": true]]))
        XCTAssertEqual(conversation.state, .listening)
        conversation.handle(YoobConversation.encode(["serverContent": ["turnComplete": true]]))
        XCTAssertEqual(conversation.state, .listening)
        conversation.send(text: "Tell me a joke")
        XCTAssertEqual(socket.sent.last as NSDictionary?, ["realtimeInput": ["text": "Tell me a joke"]] as NSDictionary)
        XCTAssertEqual(conversation.state, .thinking)
        conversation.handle(YoobConversation.encode(["serverContent": ["turnComplete": true]]))
        XCTAssertEqual(conversation.state, .listening)
    }

    func testGeminiServerErrorsGoAwayAndDisconnects() async throws {
        let socket = GeminiFakeSocket()
        let conversation = makeConversation(socket: socket)
        _ = try await conversation.open(token: "t")
        socket.server(["goAway": ["timeLeft": "30s"]])
        await settle { conversation.goAwayTimeLeft != nil }
        XCTAssertEqual(conversation.goAwayTimeLeft, "30s")
        socket.server(["error": ["code": 429, "message": "Quota exceeded"]])
        await settle { conversation.lastError != nil }
        XCTAssertEqual(conversation.lastError, .network("Quota exceeded"))
        socket.disconnect("1011: Internal error")
        await settle { conversation.state == .ended }
        XCTAssertEqual(conversation.state, .ended)
        XCTAssertEqual(conversation.lastError, .network("the conversation disconnected (1011: Internal error)"))
    }
}
