import Foundation
@preconcurrency import AVFoundation
import Observation
import LiveKit
import Yoob

/// Gives a LiveKit voice agent a Yoob face, rendered on the device. The agent needs no avatar worker and publishes no
/// video: LiveKit plays the agent's voice through its own audio engine (with its echo cancellation), and the session
/// feeds the same audio to the avatar in external-clock mode (`appendAudio` + `audioPlayed`), so the lips follow what
/// the listener hears.
///
/// ```swift
/// let session = YoobLiveKitSession(avatar: avatar, room: Room())
/// try await session.start(url: liveKitURL, token: token)   // connects and publishes the microphone
/// ```
///
/// Replies are split into utterances by the agent's `lk.agent.state` attribute (LiveKit Agents sets it), or by
/// silence when an agent does not publish it. Captions come from the `lk.transcription` text streams.
@MainActor @Observable
public final class YoobLiveKitSession {
    public enum State: Equatable, Sendable { case idle, connecting, listening, thinking, speaking, ended }

    public struct Options: Sendable {
        /// Publish the device microphone when the session starts.
        public var publishMicrophone = true
        /// Follow this participant. nil follows the first agent in the room (avatar workers are skipped).
        public var agentIdentity: String?
        /// Read captions from the `lk.transcription` text streams. Turn it off if your app registers its own handler
        /// for that topic (LiveKit allows one handler per topic).
        public var transcriptions = true
        /// Time from LiveKit rendering a buffer to the listener hearing it. nil measures it from the audio session
        /// (`outputLatency + ioBufferDuration` on iOS) each time the agent starts speaking.
        public var outputLatency: Duration?
        /// Added to the latency above. Positive values move the lips later, negative earlier.
        public var syncOffset: Duration = .zero
        /// RMS level (0–1) above which audio counts as voice. The default is about −46 dBFS.
        public var voiceThreshold = 0.005
        /// Without `lk.agent.state`: silence that ends an utterance.
        public var silenceGate: Duration = .milliseconds(600)
        /// After the agent leaves `speaking`: how long audio still in flight is followed.
        public var speakingHangover: Duration = .milliseconds(400)
        /// Audio kept from just before the agent reports `speaking`, so the first syllable is not lost.
        public var preRoll: Duration = .milliseconds(200)
        public init() {}
    }

    public private(set) var state: State = .idle
    /// What the user is saying, from the agent's transcription (final once the turn is transcribed).
    public private(set) var userTranscript = ""
    /// What the agent is saying, as it streams.
    public private(set) var assistantTranscript = ""
    /// The agent being followed, once it has joined.
    public private(set) var agentIdentity: String?
    public private(set) var lastError: YoobError?

    public let room: Room
    @ObservationIgnored private let avatar: YoobAvatar
    @ObservationIgnored private let options: Options
    @ObservationIgnored let segmenter: AgentSpeechSegmenter
    @ObservationIgnored private var observer: RoomObserver?
    @ObservationIgnored private var renderer: AgentAudioRenderer?
    @ObservationIgnored private weak var renderedTrack: RemoteAudioTrack?
    @ObservationIgnored private var pump: Task<Void, Never>?
    @ObservationIgnored private var ticker: Task<Void, Never>?
    @ObservationIgnored private var connectedRoom = false
    @ObservationIgnored private var publishedMicrophone = false
    @ObservationIgnored private var listensToTranscripts = false
    @ObservationIgnored private var agentSegment: (id: String, stream: String)?
    @ObservationIgnored private var userSegment: (id: String, stream: String)?

    static let transcriptionTopic = "lk.transcription"
    static let agentStateAttribute = "lk.agent.state"

    public init(avatar: YoobAvatar, room: Room, options: Options = Options()) {
        self.avatar = avatar
        self.room = room
        self.options = options
        var configuration = AgentSpeechSegmenter.Configuration()
        configuration.voiceThreshold = options.voiceThreshold
        configuration.silenceGate = options.silenceGate
        configuration.hangover = options.speakingHangover
        configuration.preRoll = options.preRoll
        segmenter = AgentSpeechSegmenter(sink: avatar, configuration: configuration)
        let fixed = options.outputLatency, offset = options.syncOffset
        segmenter.latency = { (fixed ?? Self.measuredOutputLatency()) + offset }
        segmenter.onActiveChanged = { [weak self] _ in self?.updateState() }
    }

    /// Prepares the character, connects (when `url` and `token` are given; otherwise the room must already be
    /// connected), publishes the microphone and starts following the agent.
    public func start(url: String? = nil, token: String? = nil) async throws {
        guard pump == nil else { return }
        state = .connecting
        lastError = nil
        do {
            try await avatar.prepare()
            startPump()
            if let url, let token, room.connectionState != .connected {
                try await room.connect(url: url, token: token)
                connectedRoom = true
            }
            guard room.connectionState == .connected else { throw YoobError.network("the LiveKit room is not connected") }
            if options.transcriptions {
                do {
                    try await room.registerTextStreamHandler(for: Self.transcriptionTopic) { [weak self] reader, identity in
                        let stream = reader.info.id
                        let segment = reader.info.attributes["lk.segment_id"] ?? stream
                        for try await text in reader where !text.isEmpty {
                            await self?.receiveTranscript(text, stream: stream, segment: segment, from: identity.stringValue)
                        }
                    }
                    listensToTranscripts = true
                } catch {
                    lastError = .unsupported("captions are off: \(error.localizedDescription)")
                }
            }
            if options.publishMicrophone {
                do { try await room.localParticipant.setMicrophone(enabled: true) }
                catch { throw YoobError.permissionDenied("The microphone could not be published: \(error.localizedDescription)") }
                publishedMicrophone = true
            }
            followAgent()
            updateState()
        } catch {
            let failure = (error as? YoobError) ?? .network(error.localizedDescription)
            await stop()
            lastError = failure
            throw failure
        }
    }

    /// Mutes or unmutes the published microphone.
    public func setMicrophoneEnabled(_ enabled: Bool) async throws {
        try await room.localParticipant.setMicrophone(enabled: enabled)
        publishedMicrophone = publishedMicrophone || enabled
    }

    /// Stops following the agent. Disconnects the room if `start` connected it; otherwise unpublishes the microphone
    /// it published. The character stays on screen.
    public func stop() async {
        pump?.cancel(); pump = nil
        ticker?.cancel(); ticker = nil
        detachRenderer()
        if let observer { room.delegates.remove(delegate: observer) }
        observer = nil
        segmenter.reset()
        agentIdentity = nil
        if listensToTranscripts { await room.unregisterTextStreamHandler(for: Self.transcriptionTopic) }
        listensToTranscripts = false
        if connectedRoom {
            await room.disconnect()
        } else if publishedMicrophone {
            _ = try? await room.localParticipant.setMicrophone(enabled: false)
        }
        connectedRoom = false
        publishedMicrophone = false
        state = .ended
    }

    // MARK: - Events

    enum Event: Sendable {
        case audio(AgentAudioChunk)
        case participantsChanged
        case disconnected(String?)
    }

    private func startPump() {
        let (events, sink) = AsyncStream<Event>.makeStream(bufferingPolicy: .unbounded)
        let observer = RoomObserver(sink)
        self.observer = observer
        room.delegates.add(delegate: observer)
        renderer = AgentAudioRenderer(sink)
        // One ordered stream, so audio and attribute changes are handled in the order they arrived.
        pump = Task { [weak self] in
            for await event in events { self?.handle(event) }
        }
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                self?.segmenter.tick(now: .now)
                try? await Task.sleep(for: .milliseconds(10))
            }
        }
    }

    func handle(_ event: Event) {
        switch event {
        case .audio(let chunk):
            segmenter.receive(chunk)
        case .participantsChanged:
            followAgent()
            updateState()
        case .disconnected(let reason):
            guard pump != nil else { return }
            if let reason { lastError = .network(reason) }
            connectedRoom = false
            Task { await self.stop() }
        }
    }

    /// Picks the agent, attaches the renderer to its voice and reads its state.
    private func followAgent() {
        let agent = room.remoteParticipants.values
            .filter { participant in
                guard let identity = participant.identity?.stringValue else { return false }
                if let wanted = options.agentIdentity { return identity == wanted }
                return participant.kind == .agent && participant.attributes["lk.publish_on_behalf"] == nil
            }
            .sorted { ($0.joinedAt ?? .distantPast) < ($1.joinedAt ?? .distantPast) }
            .first
        let identity = agent?.identity?.stringValue
        if identity != agentIdentity {
            detachRenderer()
            segmenter.reset()
            agentIdentity = identity
        }
        guard let agent else { return }
        let track = agent.audioTracks.first { $0.source == .microphone && $0.track != nil }?.track as? RemoteAudioTrack
            ?? agent.audioTracks.lazy.compactMap { $0.track as? RemoteAudioTrack }.first
        if track !== renderedTrack {
            detachRenderer()
            if let track, let renderer {
                track.add(audioRenderer: renderer)
                renderedTrack = track
            }
        }
        segmenter.setAgentState(agent.attributes[Self.agentStateAttribute], at: .now)
    }

    private func detachRenderer() {
        if let renderedTrack, let renderer { renderedTrack.remove(audioRenderer: renderer) }
        renderedTrack = nil
    }

    func updateState() {
        guard pump != nil, state != .ended else { return }
        state = Self.state(hasAgent: agentIdentity != nil, agentState: segmenter.agentState, isSpeaking: segmenter.isActive)
    }

    /// The session state for the agent's `lk.agent.state`, or for the silence gate when the agent does not publish it.
    static func state(hasAgent: Bool, agentState: String?, isSpeaking: Bool) -> State {
        guard hasAgent else { return .connecting }
        switch agentState {
        case "speaking": return .speaking
        case "thinking": return .thinking
        case "initializing": return .connecting
        case nil: return isSpeaking ? .speaking : .listening
        default: return .listening
        }
    }

    // MARK: - Captions

    func receiveTranscript(_ text: String, stream: String, segment: String, from identity: String) {
        if identity == room.localParticipant.identity?.stringValue {
            userTranscript = Self.merge(text, into: userTranscript, segment: segment, stream: stream, current: &userSegment)
        } else if agentIdentity == nil || identity == agentIdentity {
            assistantTranscript = Self.merge(text, into: assistantTranscript, segment: segment, stream: stream, current: &agentSegment)
        }
    }

    /// LiveKit Agents sends an agent's reply as chunks of one stream, and a user's turn as a new stream carrying the
    /// full text so far, both under a stable segment id. A new segment starts a new caption.
    static func merge(_ text: String, into caption: String, segment: String, stream: String,
                      current: inout (id: String, stream: String)?) -> String {
        defer { current = (segment, stream) }
        guard let current, current.id == segment else { return text }
        return current.stream == stream ? caption + text : text
    }

    nonisolated static func measuredOutputLatency() -> Duration {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        return .microseconds(Int64((session.outputLatency + session.ioBufferDuration) * 1_000_000))
        #else
        return .milliseconds(30)
        #endif
    }
}

/// Receives the agent's decoded audio from LiveKit on the audio thread.
final class AgentAudioRenderer: NSObject, AudioRenderer, @unchecked Sendable {
    private let converter = PCMConverter()
    private let sink: AsyncStream<YoobLiveKitSession.Event>.Continuation

    init(_ sink: AsyncStream<YoobLiveKitSession.Event>.Continuation) { self.sink = sink }

    func render(pcmBuffer: AVAudioPCMBuffer) {
        let arrival = ContinuousClock.now
        guard let (pcm, level) = converter.convert(pcmBuffer) else { return }
        sink.yield(.audio(AgentAudioChunk(pcm: pcm, level: level, arrival: arrival)))
    }
}

/// Forwards the room events the session needs. LiveKit holds delegates weakly; the session keeps this alive.
final class RoomObserver: NSObject, RoomDelegate, @unchecked Sendable {
    private let sink: AsyncStream<YoobLiveKitSession.Event>.Continuation

    init(_ sink: AsyncStream<YoobLiveKitSession.Event>.Continuation) { self.sink = sink }

    func room(_ room: Room, participantDidConnect participant: RemoteParticipant) { sink.yield(.participantsChanged) }
    func room(_ room: Room, participantDidDisconnect participant: RemoteParticipant) { sink.yield(.participantsChanged) }
    func room(_ room: Room, participant: Participant, didUpdateAttributes attributes: [String: String]) {
        sink.yield(.participantsChanged)
    }
    func room(_ room: Room, participant: RemoteParticipant, didSubscribeTrack publication: RemoteTrackPublication) {
        sink.yield(.participantsChanged)
    }
    func room(_ room: Room, participant: RemoteParticipant, didUnsubscribeTrack publication: RemoteTrackPublication) {
        sink.yield(.participantsChanged)
    }
    func room(_ room: Room, didDisconnectWithError error: LiveKitError?) {
        sink.yield(.disconnected(error?.localizedDescription))
    }
}
