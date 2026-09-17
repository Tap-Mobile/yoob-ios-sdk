import Foundation
import Observation

/// A spoken conversation with a Yoob character over the OpenAI Realtime protocol: either Yoob voice
/// (`init(avatar:options:voiceSession:)`, no provider key, billed through your Yoob workspace) or your own OpenAI account
/// (`init(avatar:options:clientSecret:)`). Microphone audio goes from the device to the voice service; replies stream
/// into the avatar, which plays them in sync. Speaking over the character interrupts it, and the model is told how much
/// of its reply was heard.
@MainActor @Observable
public final class YoobConversation {
    public enum State: Equatable, Sendable { case idle, connecting, listening, thinking, speaking, ended }

    public enum TurnDetection: Sendable {
        /// Commits the turn after a fixed silence. Replies start about 0.8 s sooner than `.semantic`.
        case serverVAD(silenceMS: Int = 450, threshold: Double = 0.5, prefixPaddingMS: Int = 300)
        /// Waits to judge whether the sentence is finished; tolerates mid-sentence pauses.
        case semantic(eagerness: String = "auto")
    }

    /// With Yoob voice, the voice session decides the model, speed, turn detection, noise reduction and transcription,
    /// and those options are ignored. `voice` and `instructions` apply only when your backend didn't set them.
    public struct Options: Sendable {
        public var model = "gpt-realtime"
        public var voice: String?
        public var instructions: String?
        /// Spoken speed, 0.25–1.5. 1.08 sounds natural but snappy.
        public var speed = 1.08
        /// Raise the server VAD threshold for noisy rooms instead of muting the microphone.
        public var turnDetection = TurnDetection.serverVAD()
        /// `far_field` suits a phone on a table; `near_field` suits headsets. nil turns it off.
        public var noiseReduction: String? = "far_field"
        /// Transcribes what the user says for captions. nil turns it off.
        public var transcriptionModel: String? = "gpt-4o-mini-transcribe"
        /// Have the character speak first.
        public var greet = false
        /// Hosts a Yoob voice session may connect to. The default, `*.yoob.com`, refuses a session `url` anywhere
        /// else, so a compromised or misconfigured backend can't send microphone audio elsewhere. Change it only to
        /// self-host the relay, for example `["voice.example.com"]`. `*.` matches any subdomain; only `wss` is allowed.
        public var voiceHosts: [String] = YoobConversation.defaultVoiceHosts
        public init() {}
    }

    public nonisolated static let defaultVoiceHosts = ["*.yoob.com"]

    public private(set) var state: State = .idle
    /// What the user is saying (final once the turn is transcribed).
    public private(set) var userTranscript = ""
    /// What the character is saying, as it streams.
    public private(set) var assistantTranscript = ""
    public private(set) var lastError: YoobError?

    @ObservationIgnored private let avatar: YoobAvatar
    @ObservationIgnored private let options: Options
    @ObservationIgnored private let connection: Connection
    @ObservationIgnored private let connect: @Sendable (URLRequest) -> RealtimeSocket
    @ObservationIgnored private var socket: RealtimeSocket?
    @ObservationIgnored private var receiver: Task<Void, Never>?
    @ObservationIgnored private var activeResponse: String?
    @ObservationIgnored private var playingItem: String?
    @ObservationIgnored private var finished = Set<String>()
    @ObservationIgnored private var stops = 0

    private enum Connection {
        case yoob(@Sendable () async throws -> YoobVoiceSession)
        case openAI(@Sendable () async throws -> String)
    }

    /// Talks through Yoob voice: no provider key needed, and minutes are billed through your Yoob workspace.
    /// - Parameter voiceSession: Returns a voice session from your backend, which calls
    ///   `POST https://api2.yoob.com/api/v1/voice/sessions` with your Yoob API key. A session opens one conversation and
    ///   must be used within 5 minutes, so it is requested on every `start()`.
    public convenience init(avatar: YoobAvatar, options: Options = Options(),
                            voiceSession: @escaping @Sendable () async throws -> YoobVoiceSession) {
        self.init(avatar: avatar, options: options, voiceSession: voiceSession) { URLSessionRealtimeSocket(request: $0) }
    }

    /// Talks through your own OpenAI account.
    /// - Parameter clientSecret: Returns a short-lived OpenAI Realtime client secret from your backend
    ///   (`POST https://api.openai.com/v1/realtime/client_secrets`). Never ship your OpenAI key in an app.
    public convenience init(avatar: YoobAvatar, options: Options = Options(),
                            clientSecret: @escaping @Sendable () async throws -> String) {
        self.init(avatar: avatar, options: options, clientSecret: clientSecret) { URLSessionRealtimeSocket(request: $0) }
    }

    init(avatar: YoobAvatar, options: Options, clientSecret: @escaping @Sendable () async throws -> String,
         connect: @escaping @Sendable (URLRequest) -> RealtimeSocket) {
        self.avatar = avatar; self.options = options; connection = .openAI(clientSecret); self.connect = connect
    }

    init(avatar: YoobAvatar, options: Options, voiceSession: @escaping @Sendable () async throws -> YoobVoiceSession,
         connect: @escaping @Sendable (URLRequest) -> RealtimeSocket) {
        self.avatar = avatar; self.options = options; connection = .yoob(voiceSession); self.connect = connect
    }

    private var isYoobVoice: Bool {
        if case .yoob = connection { return true }
        return false
    }

    /// Starts listening. Prepares the character, asks for the microphone and connects.
    public func start(inputID: String? = nil) async throws {
        guard socket == nil else { return }
        state = .connecting
        lastError = nil
        let generation = stops
        do {
            try await avatar.prepare()
            let socket = try await open()
            let microphone = avatar.microphone
            microphone.onAudio = { [socket] pcm in
                socket.send(Self.encode(["type": "input_audio_buffer.append", "audio": pcm.base64EncodedString()]))
            }
            try await microphone.start(inputID: inputID)
            // The socket may have closed while the microphone started (an expired grant, a quota), or stop() was called.
            guard self.socket === socket, stops == generation else {
                if let lastError { throw lastError }
                stop()
                return
            }
            state = .listening
            if options.greet { send(["type": "response.create"]) }
        } catch {
            let failure = (error as? YoobError) ?? .network(error.localizedDescription)
            stop()
            lastError = failure
            throw failure
        }
    }

    /// Fetches the credential, connects and configures the session.
    func open() async throws -> RealtimeSocket {
        let request: URLRequest
        switch connection {
        case .yoob(let voiceSession):
            // A voice session opens one connection and can't be refreshed, so it is fetched right before connecting.
            request = try Self.request(for: try await voiceSession(), allowedHosts: options.voiceHosts)
        case .openAI(let clientSecret):
            let secret = try await clientSecret()
            var components = URLComponents(string: "wss://api.openai.com/v1/realtime")!
            components.queryItems = [URLQueryItem(name: "model", value: options.model)]
            var openAI = URLRequest(url: components.url!)
            openAI.setValue("Bearer \(secret)", forHTTPHeaderField: "Authorization")
            request = openAI
        }
        let socket = connect(request)
        self.socket = socket
        socket.resume()
        receive(from: socket)
        for update in isYoobVoice ? voiceUpdates() : [sessionUpdate()] { send(update) }
        return socket
    }

    /// Sends typed text as the user's turn.
    public func send(text: String) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        bargeIn()
        send(["type": "conversation.item.create",
              "item": ["type": "message", "role": "user", "content": [["type": "input_text", "text": text]]]])
        send(["type": "response.create"])
        state = .thinking
    }

    /// Ends the conversation and releases the microphone. The character stays on screen.
    public func stop() {
        stops += 1
        receiver?.cancel(); receiver = nil
        avatar.microphone.onAudio = nil
        avatar.microphone.stop()
        avatar.interrupt()
        socket?.close()
        socket = nil
        activeResponse = nil; playingItem = nil
        state = .ended
    }

    // MARK: - Events

    private func receive(from socket: RealtimeSocket) {
        receiver = Task { [weak self] in
            while !Task.isCancelled {
                do {
                    let text = try await socket.receive()
                    self?.handle(text)
                } catch {
                    guard !Task.isCancelled, let self, self.socket === socket else { return }
                    if self.isYoobVoice, let code = socket.closeCode {
                        self.lastError = .voiceClosed(code: code)
                    } else {
                        self.lastError = .network("the conversation disconnected")
                    }
                    self.stop()
                    return
                }
            }
        }
    }

    /// Tests: talk over `socket` without starting the microphone.
    func attach(_ socket: RealtimeSocket) { self.socket = socket; state = .listening }

    func handle(_ frame: String) {
        guard let data = frame.data(using: .utf8),
              let event = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = event["type"] as? String else { return }
        let responseID = event["response_id"] as? String
        switch type {
        case "input_audio_buffer.speech_started":
            bargeIn()
            userTranscript = ""
            state = .listening
        case "input_audio_buffer.speech_stopped":
            state = .thinking
        case "conversation.item.input_audio_transcription.delta":
            userTranscript += event["delta"] as? String ?? ""
        case "conversation.item.input_audio_transcription.completed":
            userTranscript = event["transcript"] as? String ?? userTranscript
        case "response.created":
            if let id = (event["response"] as? [String: Any])?["id"] as? String, !finished.contains(id) {
                activeResponse = id
                assistantTranscript = ""
            }
        case "response.output_item.added":
            if responseID == activeResponse, let id = (event["item"] as? [String: Any])?["id"] as? String { playingItem = id }
        case "response.output_audio.delta":
            guard responseID != nil, responseID == activeResponse,
                  let base64 = event["delta"] as? String, let pcm = Data(base64Encoded: base64) else { return }
            try? avatar.speak(pcm: pcm, sampleRate: 24000)
            state = .speaking
        case "response.output_audio.done":
            if responseID == activeResponse { avatar.endSpeech() }
        case "response.output_audio_transcript.delta":
            if responseID == activeResponse { assistantTranscript += event["delta"] as? String ?? "" }
        case "response.done":
            guard let response = event["response"] as? [String: Any], let id = response["id"] as? String else { return }
            finished.insert(id)
            let status = response["status"] as? String
            if id == activeResponse {
                // A reply cut short by a limit still plays what arrived; a cancelled or failed one is dropped.
                if status != "completed" && status != "incomplete" { avatar.interrupt() }
                avatar.endSpeech()
                if state == .thinking { state = .listening }
            }
        case "error":
            let error = event["error"] as? [String: Any]
            let code = error?["code"] as? String ?? error?["type"] as? String ?? "server_error"
            if code.contains("response_cancel") || code.contains("no_active_response") { return }
            // Yoob voice: the backend already set the voice or instructions, and those win.
            if code == "yoob_voice_locked" || code == "yoob_instructions_locked" { return }
            lastError = .network(error?["message"] as? String ?? code)
        default:
            break
        }
    }

    /// The user started talking: stop the character and tell the model how much of its reply was heard.
    private func bargeIn() {
        let heardSamples = avatar.interrupt()
        if let activeResponse {
            finished.insert(activeResponse)
            send(["type": "response.cancel", "response_id": activeResponse])
            if let playingItem {
                send(["type": "conversation.item.truncate", "item_id": playingItem, "content_index": 0,
                      "audio_end_ms": heardSamples * 1000 / 24000])
            }
        }
        activeResponse = nil
        playingItem = nil
    }

    /// The Yoob voice connection: the grant rides in the Authorization header.
    static func request(for session: YoobVoiceSession, allowedHosts: [String] = defaultVoiceHosts) throws -> URLRequest {
        guard session.url.scheme?.lowercased() == "wss" else {
            throw YoobError.voiceSession(code: 0, message: "The voice session URL must use wss://.")
        }
        guard isAllowedVoiceURL(session.url, hosts: allowedHosts) else {
            throw YoobError.voiceSession(code: 0, message:
                "The voice session URL isn't on an allowed host. Yoob voice runs on *.yoob.com; set voiceHosts to self-host.")
        }
        var request = URLRequest(url: session.url)
        request.setValue("Bearer \(session.voiceToken)", forHTTPHeaderField: "Authorization")
        request.setValue("realtime", forHTTPHeaderField: "Sec-WebSocket-Protocol")
        return request
    }

    /// Whether `url` is a `wss` URL on one of `hosts` (exact names, or `*.domain` for any subdomain of it).
    nonisolated static func isAllowedVoiceURL(_ url: URL, hosts: [String]) -> Bool {
        guard url.scheme?.lowercased() == "wss", url.user == nil, url.password == nil,
              var host = url.host?.lowercased(), !host.isEmpty else { return false }
        if host.hasSuffix(".") { host.removeLast() }
        return hosts.contains { entry in
            let pattern = entry.trimmingCharacters(in: .whitespaces).lowercased()
            if pattern.hasPrefix("*.") {
                let domain = pattern.dropFirst(2)
                return !domain.isEmpty && host.hasSuffix("." + domain)
            }
            return !pattern.isEmpty && host == pattern
        }
    }

    /// Yoob voice configures the session itself. The relay accepts only instructions and voice from the app, and only
    /// when the backend didn't set them, so each goes in its own update: a refused one doesn't block the other.
    func voiceUpdates() -> [[String: Any]] {
        var updates: [[String: Any]] = []
        if let instructions = options.instructions {
            updates.append(["type": "session.update", "session": ["instructions": instructions]])
        }
        if let voice = options.voice {
            updates.append(["type": "session.update", "session": ["audio": ["output": ["voice": voice]]]])
        }
        return updates
    }

    private func sessionUpdate() -> [String: Any] {
        let turn: [String: Any]
        switch options.turnDetection {
        case .serverVAD(let silence, let threshold, let padding):
            turn = ["type": "server_vad", "silence_duration_ms": silence, "threshold": threshold,
                    "prefix_padding_ms": padding, "create_response": true, "interrupt_response": true]
        case .semantic(let eagerness):
            turn = ["type": "semantic_vad", "eagerness": eagerness, "create_response": true, "interrupt_response": true]
        }
        var input: [String: Any] = ["format": ["type": "audio/pcm", "rate": 24000], "turn_detection": turn]
        input["noise_reduction"] = options.noiseReduction.map { ["type": $0] } ?? NSNull()
        input["transcription"] = options.transcriptionModel.map { ["model": $0] } ?? NSNull()
        var output: [String: Any] = ["format": ["type": "audio/pcm", "rate": 24000], "speed": options.speed]
        if let voice = options.voice { output["voice"] = voice }
        var session: [String: Any] = ["type": "realtime", "output_modalities": ["audio"], "audio": ["input": input, "output": output]]
        if let instructions = options.instructions { session["instructions"] = instructions }
        return ["type": "session.update", "session": session]
    }

    private func send(_ event: [String: Any]) {
        socket?.send(Self.encode(event))
    }

    nonisolated static func encode(_ event: [String: Any]) -> String {
        guard let data = try? JSONSerialization.data(withJSONObject: event) else { return "{}" }
        return String(decoding: data, as: UTF8.self)
    }
}

/// The WebSocket the conversation talks over (replaceable in tests).
protocol RealtimeSocket: AnyObject, Sendable {
    func resume()
    func send(_ text: String)
    func receive() async throws -> String
    func close()
    /// Why the server closed the connection, when it said (for example "1008: token expired").
    var closeReason: String? { get }
    /// The server's close code, once the connection has closed.
    var closeCode: Int? { get }
}

extension RealtimeSocket {
    var closeReason: String? { nil }
    var closeCode: Int? { nil }
}

final class URLSessionRealtimeSocket: RealtimeSocket, @unchecked Sendable {
    private let task: URLSessionWebSocketTask
    init(request: URLRequest) { task = URLSession.shared.webSocketTask(with: request) }
    func resume() { task.resume() }
    func send(_ text: String) { task.send(.string(text)) { _ in } }
    func receive() async throws -> String {
        switch try await task.receive() {
        case .string(let text): return text
        case .data(let data): return String(decoding: data, as: UTF8.self)
        @unknown default: return ""
        }
    }
    func close() { task.cancel(with: .normalClosure, reason: nil) }
    var closeCode: Int? { task.closeCode == .invalid ? nil : task.closeCode.rawValue }
    var closeReason: String? {
        guard task.closeCode != .invalid else { return nil }
        let reason = task.closeReason.map { String(decoding: $0, as: UTF8.self) } ?? ""
        return reason.isEmpty ? "\(task.closeCode.rawValue)" : "\(task.closeCode.rawValue): \(reason)"
    }
}

/// What `POST https://api2.yoob.com/api/v1/voice/sessions` returns to your backend. Pass the response body to the app
/// unchanged and decode it with `JSONDecoder`. If the body is a Yoob API error instead, decoding throws
/// `YoobError.outOfCredit` (`402 quota_exceeded`) or `YoobError.unauthorized`.
public struct YoobVoiceSession: Sendable, Decodable, Equatable {
    /// Opens one conversation. Use it within 5 minutes.
    public let voiceToken: String
    /// The Yoob voice relay, for example `wss://voice.yoob.com/v1/realtime?model=gpt-realtime-2.1-mini`.
    public let url: URL
    public let id: String?
    public let model: String?
    public let maxSeconds: Int?
    public let creditsPerMinute: Double?
    public let expiresAt: String?

    public init(voiceToken: String, url: URL, id: String? = nil, model: String? = nil, maxSeconds: Int? = nil,
                creditsPerMinute: Double? = nil, expiresAt: String? = nil) {
        self.voiceToken = voiceToken; self.url = url; self.id = id; self.model = model
        self.maxSeconds = maxSeconds; self.creditsPerMinute = creditsPerMinute; self.expiresAt = expiresAt
    }

    enum CodingKeys: String, CodingKey {
        case voiceToken = "voice_token", url, id = "voice_session_id", model, maxSeconds = "max_seconds"
        case creditsPerMinute = "credits_per_minute", expiresAt = "expires_at", code, error
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard c.contains(.voiceToken) else {
            if (try? c.decode(String.self, forKey: .code)) == "quota_exceeded" { throw YoobError.outOfCredit }
            if c.contains(.error) || c.contains(.code) { throw YoobError.unauthorized }
            throw YoobError.voiceSession(code: 0, message: "The backend didn't return a Yoob voice session.")
        }
        self.init(voiceToken: try c.decode(String.self, forKey: .voiceToken),
                  url: try c.decode(URL.self, forKey: .url),
                  id: try c.decodeIfPresent(String.self, forKey: .id),
                  model: try c.decodeIfPresent(String.self, forKey: .model),
                  maxSeconds: try c.decodeIfPresent(Int.self, forKey: .maxSeconds),
                  creditsPerMinute: try c.decodeIfPresent(Double.self, forKey: .creditsPerMinute),
                  expiresAt: try c.decodeIfPresent(String.self, forKey: .expiresAt))
    }
}
