import Foundation
import CoreGraphics
import ImageIO
import Observation
import YoobRealistic

/// A character that speaks the audio you give it. Create one per character, call `prepare()`, show it with
/// `YoobAvatarView`, then pass speech to `speak(pcm:sampleRate:)`.
///
/// The avatar plays the audio itself so the lips stay in sync. If your app already plays the audio, use
/// `appendAudio(pcm:sampleRate:)` and report playback with `audioPlayed(samples:)` instead.
@MainActor @Observable
public final class YoobAvatar {
    public enum Phase: Equatable, Sendable {
        /// `prepare()` has not been called.
        case notPrepared
        /// Character files are downloading. The poster and idle frames appear before the models finish.
        case downloading(YoobProgress)
        /// Files are ready; the renderer is warming up (well under a second once compiled).
        case warming
        /// Ready to speak. Idle frames play.
        case ready
        /// Speech is playing and the face follows it.
        case speaking
        /// Preparing failed. `prepare()` can be called again.
        case failed(YoobError)
        /// The session was stopped, for example because the workspace ran out of credit.
        case stopped(YoobError)
    }

    public private(set) var phase: Phase = .notPrepared
    /// The character's still image, available early in the download.
    public private(set) var poster: CGImage?
    /// Closed-mouth frames played forward and back while the character is silent.
    public private(set) var idleFrames: [CGImage] = []
    public private(set) var idleFramesPerSecond: Double = 8
    /// The current speaking frame. Shown instead of the idle frames while `isShowingSpeech` is true.
    public private(set) var speechFrame: CGImage?
    public private(set) var isShowingSpeech = false
    /// Width over height of every frame this character draws.
    public private(set) var aspectRatio: CGFloat = 9.0 / 16.0
    public private(set) var manifest: CharacterManifest?
    /// The character id this avatar was created for (nil for a local pack until it is opened).
    public var character: String? {
        if case .cloud(let character, _) = source { return character }
        return manifest?.character
    }
    /// Sync diagnostics: frames shown and frames skipped because rendering fell behind the audio.
    public private(set) var stats = Stats()
    /// Why the renderer stopped during the last utterance, if it did. The audio kept playing.
    public private(set) var lastRendererError: String?
    public struct Stats: Equatable, Sendable {
        public var framesShown = 0
        public var framesSkipped = 0
        public var utterances = 0
    }

    /// Longest the voice is held back at the start of an utterance while the first frame renders.
    public var maxSyncDelay: Duration = .milliseconds(1800)

    private let source: YoobSource
    private let version: String?
    private var credentials: YoobCredentials?
    private var engine: FaceEngine?
    private var packDirectory: URL?
    private var preparing: Task<Void, Error>?
    private let player = SpeechPlayer()
    private var utterance: Utterance?
    private var heartbeat: Task<Void, Never>?
    /// Call frame the head reached; the next utterance continues from there.
    private var hostFrame = 0
    private var externalClock = false

    private final class Utterance {
        let id: Int
        let sampleRate: Double
        let resampler: AvatarResampler
        let feed: AsyncStream<[Float]>.Continuation
        let consumer: Task<Void, Never>
        var held: [Data] = []
        var heldSince: ContinuousClock.Instant?
        var started = false
        var ended = false
        var engineFailed = false
        var receivedSamples: Int64 = 0
        var externalPlayed: Int64 = 0
        var prepared: [Int: CGImage] = [:]
        var nextToPrepare = 0
        var preparing = false
        var shown = -1
        init(id: Int, sampleRate: Double, resampler: AvatarResampler, feed: AsyncStream<[Float]>.Continuation, consumer: Task<Void, Never>) {
            self.id = id; self.sampleRate = sampleRate; self.resampler = resampler; self.feed = feed; self.consumer = consumer
        }
    }
    private var utteranceCount = 0
    private var ticker: Task<Void, Never>?

    public init(_ source: YoobSource, version: String? = nil) {
        self.source = source
        self.version = version
        player.onDrained = { [weak self] in Task { @MainActor in self?.playbackDrained() } }
    }

    // MARK: - Preparing

    /// Downloads (or opens) the character and warms up the renderer. Safe to call again after a failure; a finished
    /// download is reused.
    public func prepare() async throws {
        if let preparing { return try await preparing.value }
        if case .ready = phase { return }
        if case .speaking = phase { return }
        let task = Task { try await self.load() }
        preparing = task
        defer { preparing = nil }
        do { try await task.value }
        catch {
            let failure: YoobError
            if let known = error as? YoobError { failure = known }
            else if error is CocoaError || error is URLError { failure = .network(error.localizedDescription) }
            else { failure = .renderer(error.localizedDescription) }
            if !(error is CancellationError) { phase = .failed(failure) }
            throw failure
        }
    }

    private func load() async throws {
        let manifest: CharacterManifest
        let root: URL
        switch source {
        case .local(let directory):
            let data: Data
            do { data = try Data(contentsOf: directory.appendingPathComponent("character.json")) }
            catch { throw YoobError.invalidAssets("character.json missing in \(directory.lastPathComponent)") }
            manifest = try JSONDecoder().decode(CharacterManifest.self, from: data)
            try manifest.validate()
            root = directory
            show(manifest: manifest, root: root)
        case .cloud(let character, let fetchCredentials):
            phase = .downloading(YoobProgress(completedBytes: 0, totalBytes: 0))
            let credentials = try await fetchCredentials()
            self.credentials = credentials
            let store = AssetStore.shared
            do {
                manifest = try await store.manifest(character: character, version: version, credentials: credentials)
            } catch let error as YoobError {
                // Offline with a complete pack on disk: start from it.
                guard case .network = error, version == nil, let cached = await store.cachedManifest(character: character) else { throw error }
                manifest = cached
            }
            let progress: @Sendable (YoobProgress) -> Void = { value in
                Task { @MainActor [weak self] in
                    guard let self, case .downloading = self.phase else { return }
                    self.phase = .downloading(value)
                }
            }
            // The poster and idle frames first, so the character is on screen while the models download.
            let early = try await store.download(manifest, throughTier: 1, credentials: credentials, progress: { _ in })
            show(manifest: manifest, root: early)
            root = try await store.download(manifest, throughTier: 3, credentials: credentials, progress: progress)
            startHeartbeat()
        }
        packDirectory = root
        phase = .warming
        engine = try await FaceEngines.load(manifest, root: root)
        phase = .ready
    }

    private func show(manifest: CharacterManifest, root: URL) {
        self.manifest = manifest
        aspectRatio = CGFloat(manifest.width) / CGFloat(manifest.height)
        idleFramesPerSecond = manifest.idle.fps
        poster = Self.decode(root.appendingPathComponent(manifest.poster))
        let urls = manifest.idle.frames.map { root.appendingPathComponent($0) }
        Task {
            let frames = await Task.detached(priority: .userInitiated) { urls.compactMap(Self.decode) }.value
            self.idleFrames = frames
        }
    }

    nonisolated private static func decode(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }

    // MARK: - Speaking

    /// Plays mono 16-bit little-endian PCM and moves the face with it. Call repeatedly as audio streams in; call
    /// `endSpeech()` after the last chunk. Audio plays even if the character is not ready yet.
    public func speak(pcm: Data, sampleRate: Int = 24000) throws {
        externalClock = false
        try ingest(pcm, sampleRate: sampleRate)
    }

    /// For apps that play the audio themselves: renders frames for this audio without playing it. Report playback
    /// progress with `audioPlayed(samples:)`.
    public func appendAudio(pcm: Data, sampleRate: Int = 24000) throws {
        externalClock = true
        try ingest(pcm, sampleRate: sampleRate)
    }

    /// With `appendAudio`: how many samples of the current utterance the listener has heard.
    public func audioPlayed(samples: Int) {
        guard let utterance else { return }
        utterance.externalPlayed = max(utterance.externalPlayed, Int64(samples))
        if utterance.ended, utterance.externalPlayed >= utterance.receivedSamples { finishUtterance() }
    }

    /// Marks the end of the current utterance. The face returns to idle when the audio finishes.
    public func endSpeech() {
        guard let utterance, !utterance.ended else { return }
        utterance.ended = true
        if let engine { utterance.feed.yield([Float](repeating: 0, count: engine.tailSamples)) }
        utterance.feed.finish()
        releaseHeldAudio(force: true)
        if externalClock, utterance.externalPlayed >= utterance.receivedSamples { finishUtterance() }
    }

    /// Stops speaking at once (barge-in). Returns how many samples of the utterance were heard, which is what a
    /// realtime model needs to truncate its reply.
    @discardableResult
    public func interrupt() -> Int {
        guard let utterance else { return 0 }
        let heard = externalClock ? utterance.externalPlayed : player.playedSamples
        player.stop()
        finishUtterance()
        return Int(heard)
    }

    private func ingest(_ pcm: Data, sampleRate: Int) throws {
        guard pcm.count % 2 == 0 else { throw YoobError.invalidAudio("PCM16 data must have an even byte count") }
        if case .stopped(let reason) = phase { throw reason }
        guard !pcm.isEmpty else { return }
        let current = try utterance ?? beginUtterance(sampleRate: Double(sampleRate))
        guard current.sampleRate == Double(sampleRate) else {
            throw YoobError.invalidAudio("sample rate changed mid-utterance; call endSpeech() first")
        }
        if current.ended { finishUtterance(); return try ingest(pcm, sampleRate: sampleRate) }
        current.receivedSamples += Int64(pcm.count / 2)
        if engine != nil, !current.engineFailed {
            let bytesPerSecond = sampleRate * 2
            var offset = 0
            while offset < pcm.count {
                let slice = pcm.subdata(in: offset..<min(pcm.count, offset + bytesPerSecond))
                if let samples = try? current.resampler.convert(slice) { current.feed.yield(samples) }
                offset += slice.count
            }
        }
        guard !externalClock else { return }
        if current.started {
            player.schedule(pcm)
        } else {
            current.held.append(pcm)
            if current.heldSince == nil { current.heldSince = .now }
            releaseHeldAudio(force: engine == nil || current.engineFailed)
        }
    }

    private func beginUtterance(sampleRate: Double) throws -> Utterance {
        let resampler: AvatarResampler
        do { resampler = try AvatarResampler(sourceRate: sampleRate) }
        catch { throw YoobError.invalidAudio("unsupported sample rate \(Int(sampleRate)) Hz") }
        if !externalClock { try player.begin(sampleRate: sampleRate) }
        utteranceCount += 1
        let id = utteranceCount
        let (stream, feed) = AsyncStream<[Float]>.makeStream(bufferingPolicy: .unbounded)
        let engine = engine
        let startFrame = hostFrame
        let consumer = Task.detached(priority: .userInitiated) { [weak self] in
            guard let engine else { return }
            await engine.restart(hostFrame: startFrame)
            do {
                for await samples in stream {
                    try Task.checkCancellation()
                    try await engine.append(samples)
                    await self?.tick()
                }
            } catch is CancellationError {
            } catch {
                await self?.engineFailed(id: id, error)
            }
        }
        let utterance = Utterance(id: id, sampleRate: sampleRate, resampler: resampler, feed: feed, consumer: consumer)
        self.utterance = utterance
        stats.utterances += 1
        if case .ready = phase { phase = .speaking }
        startTicker()
        return utterance
    }

    /// Starts the voice once the first frame is ready, or after `maxSyncDelay`, so the lips start with the voice.
    private func releaseHeldAudio(force: Bool) {
        guard let utterance, !utterance.started, !externalClock else { return }
        let waited = utterance.heldSince.map { $0.duration(to: .now) } ?? .zero
        guard force || utterance.prepared[0] != nil || waited >= maxSyncDelay else { return }
        utterance.started = true
        for chunk in utterance.held { player.schedule(chunk) }
        utterance.held = []
    }

    private func engineFailed(id: Int, _ error: Error) {
        guard let utterance, utterance.id == id else { return }
        lastRendererError = String(describing: error)
        utterance.engineFailed = true
        isShowingSpeech = false
        releaseHeldAudio(force: true)
    }

    private func playbackDrained() {
        guard let utterance, utterance.ended, utterance.held.isEmpty, player.isIdle else { return }
        finishUtterance()
    }

    private func finishUtterance() {
        guard let utterance else { return }
        utterance.consumer.cancel()
        utterance.feed.finish()
        if utterance.shown >= 0 { hostFrame += utterance.shown + 1 }
        self.utterance = nil
        isShowingSpeech = false
        ticker?.cancel(); ticker = nil
        if case .speaking = phase { phase = engine == nil ? .notPrepared : .ready }
        if engine != nil, case .notPrepared = phase { phase = .ready }
    }

    // MARK: - Frame clock

    private func startTicker() {
        guard ticker == nil else { return }
        ticker = Task { [weak self] in
            while !Task.isCancelled {
                self?.tick()
                try? await Task.sleep(for: .milliseconds(8))
            }
        }
    }

    private func tick() {
        guard let utterance, let engine, !utterance.engineFailed else { return }
        if !utterance.started, !externalClock { releaseHeldAudio(force: false) }
        let played = externalClock ? utterance.externalPlayed : (utterance.started ? player.playedSamples : 0)
        let playing = externalClock ? played > 0 : utterance.started
        // Frame n covers audio from n/25 s; show it once that audio is audible.
        let playedFrame = playing ? Int(Double(played) * 25 / utterance.sampleRate) : -1

        if playedFrame >= 0, let frame = utterance.prepared.keys.filter({ $0 <= playedFrame && $0 > utterance.shown }).max(),
           let image = utterance.prepared[frame] {
            stats.framesSkipped += max(0, frame - utterance.shown - 1)
            stats.framesShown += 1
            speechFrame = image
            utterance.shown = frame
            isShowingSpeech = true
            utterance.prepared = utterance.prepared.filter { $0.key > frame }
        }
        // Frames the audio has already passed are not worth rendering.
        if utterance.nextToPrepare < playedFrame - 1 { utterance.nextToPrepare = playedFrame - 1 }
        guard !utterance.preparing, utterance.nextToPrepare <= max(playedFrame, 0) + 4 else { return }
        utterance.preparing = true
        let wanted = utterance.nextToPrepare, id = utterance.id
        Task { [weak self] in
            let image: CGImage?
            do { image = try await engine.frame(wanted) } catch { self?.engineFailed(id: id, error); return }
            guard let self, let current = self.utterance, current.id == id else { return }
            current.preparing = false
            if let image {
                current.prepared[wanted] = image
                current.nextToPrepare = wanted + 1
                self.tick()
            }
        }
    }

    // MARK: - Session

    /// Ends the metered session and releases the renderer. The downloaded files stay cached.
    public func close() async {
        interrupt()
        heartbeat?.cancel(); heartbeat = nil
        player.shutdown()
        engine = nil
        if let packDirectory { await AssetStore.shared.release(packDirectory) }
        if let credentials { _ = try? await SessionAPI.post("end", credentials: credentials) }
        credentials = nil
        phase = .notPrepared
    }

    private func startHeartbeat() {
        heartbeat?.cancel()
        heartbeat = Task { [weak self] in
            while !Task.isCancelled {
                guard let seconds = self?.credentials?.heartbeatSeconds else { return }
                try? await Task.sleep(for: .seconds(seconds))
                guard !Task.isCancelled, let self else { return }
                await self.beat()
            }
        }
    }

    private func beat() async {
        guard let credentials else { return }
        do {
            let reply = try await SessionAPI.post("heartbeat", credentials: credentials)
            guard reply.stop else { return }
            if reply.reason == "out-of-credits" { stop(.outOfCredit) } else { try await renewSession() }
        } catch YoobError.unauthorized {
            // The console ends sessions that missed beats (the app was in the background): open a new one.
            try? await renewSession()
        } catch {
            // A missed beat is retried on the next one.
        }
    }

    private func renewSession() async throws {
        guard case .cloud(_, let fetchCredentials) = source else { return }
        credentials = try await fetchCredentials()
    }

    private func stop(_ reason: YoobError) {
        interrupt()
        heartbeat?.cancel(); heartbeat = nil
        phase = .stopped(reason)
    }
}

enum SessionAPI {
    struct Reply: Decodable {
        let stop: Bool
        let reason: String?
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            stop = try c.decodeIfPresent(Bool.self, forKey: .stop) ?? false
            reason = try c.decodeIfPresent(String.self, forKey: .reason)
        }
        enum CodingKeys: String, CodingKey { case stop, reason }
    }

    static func post(_ action: String, credentials: YoobCredentials) async throws -> Reply {
        var request = URLRequest(url: credentials.apiBase.appendingPathComponent("api/v1/sessions/\(action)"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(credentials.sessionToken)", forHTTPHeaderField: "Authorization")
        request.setValue("yoob-ios/\(Yoob.version)", forHTTPHeaderField: "X-Yoob-SDK")
        request.timeoutInterval = 15
        let (data, response) = try await URLSession.shared.data(for: request)
        switch (response as? HTTPURLResponse)?.statusCode ?? 0 {
        case 200:
            if let reply = try? JSONDecoder().decode(Reply.self, from: data) { return reply }
            return try JSONDecoder().decode(Reply.self, from: Data("{}".utf8))
        case 401, 404: throw YoobError.unauthorized
        case 402: throw YoobError.outOfCredit
        case let status: throw YoobError.network("HTTP \(status)")
        }
    }
}
