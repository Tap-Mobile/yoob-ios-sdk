import Foundation
import Observation
@preconcurrency import AVFoundation

/// A spoken conversation with a Yoob character, using the Gemini Live API. Microphone audio goes from the device to
/// Gemini at 16 kHz; replies (24 kHz) stream into the avatar, which plays them in sync. Speaking over the character
/// interrupts it.
@MainActor @Observable
public final class YoobGeminiConversation {
    public typealias State = YoobConversation.State

    /// Gemini's voice activity detection.
    public struct ActivityDetection: Sendable {
        public enum Sensitivity: String, Sendable { case high = "HIGH", low = "LOW" }
        /// How readily speech is detected. `.high` (Gemini's default) lets the user interrupt quickly; use `.low` in
        /// noisy rooms instead of muting the microphone.
        public var startSensitivity = Sensitivity.high
        /// How readily the user's turn is judged finished. `.high` ends turns sooner.
        public var endSensitivity = Sensitivity.high
        /// Speech needed before a turn starts.
        public var prefixPaddingMS = 100
        /// Silence that ends a turn. 450 ms is the window Yoob measured as fastest for OpenAI Realtime.
        public var silenceMS = 450
        public init() {}
    }

    public struct Options: Sendable {
        /// Google's recommended low-latency native-audio Live model.
        public var model = "gemini-3.8-live"
        /// Prebuilt voice name, for example "Kore" or "Puck". nil lets Gemini choose.
        public var voice: String?
        public var systemInstruction: String?
        /// Have the character speak first, prompted with `greeting`.
        public var greet = false
        public var greeting = "The user just joined. Greet them briefly."
        /// Transcribe what the user says (for captions).
        public var inputTranscription = true
        /// Transcribe what the character says.
        public var outputTranscription = true
        public var activityDetection = ActivityDetection()
        public init() {}
    }

    static let endpoint =
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained"

    public private(set) var state: State = .idle
    /// What the user is saying. A new utterance replaces it once the character has answered.
    public private(set) var userTranscript = ""
    /// What the character is saying, as it streams.
    public private(set) var assistantTranscript = ""
    public private(set) var lastError: YoobError?
    /// Set when Gemini warns it will close the connection soon, for example "30s".
    public private(set) var goAwayTimeLeft: String?

    @ObservationIgnored private let avatar: YoobAvatar
    @ObservationIgnored private let options: Options
    @ObservationIgnored private let token: @Sendable () async throws -> String
    @ObservationIgnored private let makeSocket: @Sendable (URLRequest) -> RealtimeSocket
    @ObservationIgnored private var socket: RealtimeSocket?
    @ObservationIgnored private var receiver: Task<Void, Never>?
    @ObservationIgnored private var setupWaiter: CheckedContinuation<Void, Error>?
    @ObservationIgnored private var setupTimeout: Task<Void, Never>?
    @ObservationIgnored private var playbackWatch: Task<Void, Never>?
    @ObservationIgnored private var replying = false
    @ObservationIgnored private var userTurnDone = true

    /// - Parameter token: Returns a Gemini Live ephemeral token from your backend: the `name` of
    ///   `client.authTokens.create()` (`POST https://generativelanguage.googleapis.com/v1beta/auth_tokens`).
    ///   Never ship your Gemini API key in an app.
    public convenience init(avatar: YoobAvatar, options: Options = Options(),
                            token: @escaping @Sendable () async throws -> String) {
        self.init(avatar: avatar, options: options, token: token) { URLSessionRealtimeSocket(request: $0) }
    }

    init(avatar: YoobAvatar, options: Options, token: @escaping @Sendable () async throws -> String,
         connect: @escaping @Sendable (URLRequest) -> RealtimeSocket) {
        self.avatar = avatar; self.options = options; self.token = token; makeSocket = connect
    }

    /// Starts listening. Prepares the character, connects, and asks for the microphone.
    public func start(inputID: String? = nil) async throws {
        guard socket == nil else { return }
        state = .connecting
        lastError = nil
        goAwayTimeLeft = nil
        do {
            try await avatar.prepare()
            let socket = try await open(token: try await token())
            let uplink = try GeminiAudioUplink(socket: socket)
            let microphone = avatar.microphone
            microphone.onAudio = { pcm in uplink.send(pcm) }
            try await microphone.start(inputID: inputID)
            state = .listening
            if options.greet {
                send(["clientContent": ["turns": [["role": "user", "parts": [["text": options.greeting]]]], "turnComplete": true]])
                state = .thinking
            }
        } catch {
            let failure = (error as? YoobError) ?? .network(error.localizedDescription)
            stop()
            lastError = failure
            throw failure
        }
    }

    /// Sends typed text as the user's turn.
    public func send(text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        avatar.interrupt()
        finishReply()
        send(["realtimeInput": ["text": text]])
        state = .thinking
    }

    /// Ends the conversation and releases the microphone. The character stays on screen.
    public func stop() {
        receiver?.cancel(); receiver = nil
        playbackWatch?.cancel(); playbackWatch = nil
        failSetup(.network("the conversation was stopped"))
        avatar.microphone.onAudio = nil
        avatar.microphone.stop()
        avatar.interrupt()
        socket?.close()
        socket = nil
        replying = false
        state = .ended
    }

    // MARK: - Connection

    static func request(token: String) -> URLRequest {
        var allowed = CharacterSet.alphanumerics
        allowed.insert(charactersIn: "-._~")
        let encoded = token.addingPercentEncoding(withAllowedCharacters: allowed) ?? token
        return URLRequest(url: URL(string: "\(endpoint)?access_token=\(encoded)")!)
    }

    /// Connects, sends the setup and waits for Gemini to accept it.
    func open(token: String) async throws -> RealtimeSocket {
        let socket = makeSocket(Self.request(token: token))
        self.socket = socket
        socket.resume()
        receive(from: socket)
        send(["setup": setupMessage()])
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            setupWaiter = continuation
            setupTimeout = Task { [weak self] in
                try? await Task.sleep(for: .seconds(15))
                guard !Task.isCancelled else { return }
                self?.failSetup(.network("Gemini Live didn't answer in time"))
            }
        }
        return socket
    }

    private func finishSetup() {
        setupTimeout?.cancel(); setupTimeout = nil
        setupWaiter?.resume()
        setupWaiter = nil
    }

    private func failSetup(_ error: YoobError) {
        setupTimeout?.cancel(); setupTimeout = nil
        guard let waiter = setupWaiter else { return }
        setupWaiter = nil
        waiter.resume(throwing: error)
    }

    private func receive(from socket: RealtimeSocket) {
        receiver = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    let text = try await socket.receive()
                    self?.handle(text)
                } catch {
                    guard !Task.isCancelled, let self, self.socket === socket else { return }
                    let reason = socket.closeReason.map { " (\($0))" } ?? ""
                    if self.setupWaiter != nil {
                        self.failSetup(.network("Gemini Live closed the connection\(reason)"))
                    } else {
                        self.lastError = .network("the conversation disconnected\(reason)")
                        self.stop()
                    }
                    return
                }
            }
        }
    }

    func setupMessage() -> [String: Any] {
        var generation: [String: Any] = ["responseModalities": ["AUDIO"]]
        if let voice = options.voice {
            generation["speechConfig"] = ["voiceConfig": ["prebuiltVoiceConfig": ["voiceName": voice]]]
        }
        let vad = options.activityDetection
        var setup: [String: Any] = [
            "model": options.model.hasPrefix("models/") ? options.model : "models/\(options.model)",
            "generationConfig": generation,
            "realtimeInputConfig": [
                "automaticActivityDetection": [
                    "disabled": false,
                    "startOfSpeechSensitivity": "START_SENSITIVITY_\(vad.startSensitivity.rawValue)",
                    "endOfSpeechSensitivity": "END_SENSITIVITY_\(vad.endSensitivity.rawValue)",
                    "prefixPaddingMs": vad.prefixPaddingMS,
                    "silenceDurationMs": vad.silenceMS,
                ] as [String: Any],
                "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
            ] as [String: Any],
        ]
        if let instruction = options.systemInstruction { setup["systemInstruction"] = ["parts": [["text": instruction]]] }
        if options.inputTranscription { setup["inputAudioTranscription"] = [String: Any]() }
        if options.outputTranscription { setup["outputAudioTranscription"] = [String: Any]() }
        return setup
    }

    // MARK: - Events

    /// Tests: talk over `socket` without starting the microphone.
    func attach(_ socket: RealtimeSocket) { self.socket = socket; state = .listening }

    func handle(_ frame: String) {
        guard let data = frame.data(using: .utf8),
              let message = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        if message["setupComplete"] != nil {
            finishSetup()
            return
        }
        if let away = message["goAway"] as? [String: Any] {
            goAwayTimeLeft = away["timeLeft"] as? String ?? ""
            return
        }
        if let error = message["error"] as? [String: Any] {
            let text = error["message"] as? String ?? error["status"] as? String ?? "\(error["code"] ?? "server error")"
            if setupWaiter != nil { failSetup(.network(text)) } else { lastError = .network(text) }
            return
        }
        guard let content = message["serverContent"] as? [String: Any] else { return }

        if content["interrupted"] as? Bool == true {
            // The user spoke over the reply; Gemini has already cancelled it.
            avatar.interrupt()
            finishReply()
            state = .listening
        }
        if let heard = (content["inputTranscription"] as? [String: Any])?["text"] as? String, !heard.isEmpty {
            if userTurnDone { userTranscript = ""; userTurnDone = false }
            userTranscript += heard
        }
        for part in (content["modelTurn"] as? [String: Any])?["parts"] as? [[String: Any]] ?? [] {
            guard let blob = part["inlineData"] as? [String: Any], let mime = blob["mimeType"] as? String,
                  mime.hasPrefix("audio/pcm"), let base64 = blob["data"] as? String,
                  let pcm = Data(base64Encoded: base64) else { continue }
            beginReply()
            try? avatar.speak(pcm: pcm, sampleRate: Self.sampleRate(mime))
            state = .speaking
        }
        if let said = (content["outputTranscription"] as? [String: Any])?["text"] as? String, !said.isEmpty {
            beginReply()
            assistantTranscript += said
        }
        if content["generationComplete"] as? Bool == true { avatar.endSpeech() }
        if content["turnComplete"] as? Bool == true {
            avatar.endSpeech()
            finishReply()
            userTurnDone = true
            if state == .thinking { state = .listening }
            if state == .speaking { watchPlaybackEnd() }
        }
    }

    /// The first audio or text of a reply: the user's turn is over.
    private func beginReply() {
        guard !replying else { return }
        replying = true
        playbackWatch?.cancel(); playbackWatch = nil
        assistantTranscript = ""
        userTurnDone = true
    }

    private func finishReply() { replying = false }

    /// Gemini sends nothing when the reply finishes playing, so follow the avatar back to listening.
    private func watchPlaybackEnd() {
        playbackWatch?.cancel()
        playbackWatch = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, !self.replying, self.state == .speaking else { return }
                if self.avatar.phase != .speaking { self.state = .listening; return }
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }

    static func sampleRate(_ mimeType: String) -> Int {
        guard let range = mimeType.range(of: "rate=") else { return 24000 }
        return Int(mimeType[range.upperBound...].prefix { $0.isNumber }) ?? 24000
    }

    private func send(_ message: [String: Any]) {
        socket?.send(YoobConversation.encode(message))
    }
}

/// Converts microphone packets (24 kHz) to Gemini's 16 kHz and sends them. Runs on the audio thread.
final class GeminiAudioUplink: @unchecked Sendable {
    private let socket: RealtimeSocket
    private let resampler: PCM16Resampler
    private let lock = NSLock()

    init(socket: RealtimeSocket) throws {
        self.socket = socket
        resampler = try PCM16Resampler(from: 24000, to: 16000)
    }

    func send(_ pcm: Data) {
        guard let converted = lock.withLock({ try? resampler.convert(pcm) }), !converted.isEmpty else { return }
        socket.send(YoobConversation.encode(
            ["realtimeInput": ["audio": ["data": converted.base64EncodedString(), "mimeType": "audio/pcm;rate=16000"]]]))
    }
}

/// Stateful mono PCM16 sample-rate conversion, so a stream cut into packets converts without seams.
final class PCM16Resampler {
    private let source: AVAudioFormat
    private let destination: AVAudioFormat
    private let converter: AVAudioConverter
    private let ratio: Double

    init(from sourceRate: Double, to targetRate: Double) throws {
        guard let source = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sourceRate, channels: 1, interleaved: true),
              let destination = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: targetRate, channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: source, to: destination) else {
            throw YoobError.invalidAudio("can't convert \(sourceRate) Hz audio to \(targetRate) Hz")
        }
        converter.primeMethod = .none
        converter.sampleRateConverterQuality = AVAudioQuality.high.rawValue
        self.source = source; self.destination = destination; self.converter = converter
        ratio = targetRate / sourceRate
    }

    func convert(_ pcm: Data) throws -> Data {
        guard pcm.count % 2 == 0 else { throw YoobError.invalidAudio("PCM16 data must have an even byte count") }
        guard !pcm.isEmpty else { return Data() }
        let frames = AVAudioFrameCount(pcm.count / 2)
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames),
              let samples = input.int16ChannelData?[0],
              let output = AVAudioPCMBuffer(pcmFormat: destination,
                                            frameCapacity: AVAudioFrameCount(ceil(Double(frames) * ratio)) + 64) else {
            throw YoobError.invalidAudio("couldn't allocate audio buffers")
        }
        input.frameLength = frames
        pcm.withUnsafeBytes { bytes in
            for i in 0..<Int(frames) {
                samples[i] = Int16(littleEndian: bytes.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))
            }
        }
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, state in
            if supplied { state.pointee = .noDataNow; return nil }
            supplied = true
            state.pointee = .haveData
            return input
        }
        guard status != .error, error == nil, let converted = output.int16ChannelData?[0] else {
            throw YoobError.invalidAudio(error?.localizedDescription ?? "sample-rate conversion failed")
        }
        var result = Data(count: Int(output.frameLength) * 2)
        result.withUnsafeMutableBytes { bytes in
            for i in 0..<Int(output.frameLength) {
                bytes.storeBytes(of: converted[i].littleEndian, toByteOffset: i * 2, as: Int16.self)
            }
        }
        return result
    }
}
