import Foundation
@preconcurrency import AVFoundation

/// Plays one utterance's PCM16 audio and reports how much of it has actually been heard. The clock stops while the
/// buffer is empty (a slow network), so the face never runs ahead of the voice.
final class SpeechPlayer: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let lock = NSLock()
    private var format: AVAudioFormat?
    /// Samples of this utterance whose buffers finished playing, and those scheduled but not finished.
    private var finishedSamples: Int64 = 0
    private var queued: [Int64] = []
    /// Content position at which the player node last started, and whether it is running.
    private var runBase: Int64 = 0
    private var running = false
    private var generation = 0
    private let control = DispatchQueue(label: "com.yoob.speech-player")
    var onDrained: (() -> Void)?

    init() {
        engine.attach(player)
    }

    var sampleRate: Double { lock.withLock { format?.sampleRate ?? 24000 } }

    /// Samples heard so far in this utterance.
    var playedSamples: Int64 {
        lock.lock(); defer { lock.unlock() }
        guard running, let time = player.lastRenderTime.flatMap({ player.playerTime(forNodeTime: $0) }) else {
            return finishedSamples
        }
        let limit = finishedSamples + (queued.first ?? 0)
        return min(limit, runBase + max(0, time.sampleTime))
    }

    func begin(sampleRate: Double) throws {
        stop()
        lock.lock(); defer { lock.unlock() }
        if format?.sampleRate != sampleRate || !engine.isRunning {
            guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false) else {
                throw YoobError.invalidAudio("sample rate \(sampleRate)")
            }
            if capture == nil { try Self.activatePlaybackSession() }
            engine.stop()
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            self.format = format
            engine.prepare()
            do { try engine.start() } catch { throw YoobError.renderer("audio output: \(error.localizedDescription)") }
        }
        finishedSamples = 0; queued = []; runBase = 0; running = false
    }

    // MARK: - Microphone

    /// Called on the audio thread with 24 kHz mono PCM16 and its RMS level (0–1).
    typealias CaptureHandler = @Sendable (Data, Double) -> Void
    private var capture: (handler: CaptureHandler, sink: AVAudioSinkNode, clock: AVAudioSourceNode)?
    private var configurationObserver: NSObjectProtocol?
    var onCaptureFailed: (@Sendable (String) -> Void)?

    var isCapturing: Bool { lock.withLock { capture != nil } }

    /// Captures the microphone through the same engine that plays speech, with voice processing on, so the system echo
    /// canceller removes the character's voice from what is captured.
    func startCapture(_ handler: @escaping CaptureHandler) throws {
        lock.lock(); defer { lock.unlock() }
        guard capture == nil else { return }
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
            try session.setActive(true)
        } catch { throw YoobError.renderer("audio session: \(error.localizedDescription)") }
        #endif
        engine.stop()
        do { try engine.inputNode.setVoiceProcessingEnabled(true) }
        catch { throw YoobError.unsupported("echo cancellation is unavailable: \(error.localizedDescription)") }
        let output = format ?? AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24000, channels: 1, interleaved: false)!
        if format == nil {
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: output)
            format = output
        }
        // Keeps the voice-processing graph rendering between replies: speaker silence, never fake microphone input.
        let clock = AVAudioSourceNode(format: output) { @Sendable isSilence, _, _, buffers in
            isSilence.pointee = true
            for buffer in UnsafeMutableAudioBufferListPointer(buffers) {
                if let data = buffer.mData { memset(data, 0, Int(buffer.mDataByteSize)) }
            }
            return noErr
        }
        engine.attach(clock)
        engine.connect(clock, to: engine.mainMixerNode, format: output)
        let sink = try makeSink(handler)
        capture = (handler, sink, clock)
        engine.prepare()
        do { try engine.start() } catch {
            teardownCapture()
            throw YoobError.renderer("microphone: \(error.localizedDescription)")
        }
        configurationObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: nil
        ) { [weak self] _ in
            // The system stops the engine when the hardware format changes (for example a headset connecting).
            self?.control.async { self?.recoverCapture() }
        }
    }

    func stopCapture() {
        lock.lock(); defer { lock.unlock() }
        guard capture != nil else { return }
        engine.stop()
        teardownCapture()
        try? engine.inputNode.setVoiceProcessingEnabled(false)
        if let format { engine.disconnectNodeOutput(player); engine.connect(player, to: engine.mainMixerNode, format: format) }
        #if os(iOS)
        try? Self.activatePlaybackSession(force: true)
        #endif
        engine.prepare()
        try? engine.start()
    }

    func setInputMuted(_ muted: Bool) {
        engine.inputNode.isVoiceProcessingInputMuted = muted
    }

    private func teardownCapture() {
        if let observer = configurationObserver { NotificationCenter.default.removeObserver(observer) }
        configurationObserver = nil
        if let capture {
            engine.disconnectNodeOutput(engine.inputNode)
            engine.detach(capture.sink)
            engine.detach(capture.clock)
        }
        capture = nil
    }

    private func recoverCapture() {
        lock.lock(); defer { lock.unlock() }
        guard let current = capture, !engine.isRunning else { return }
        engine.disconnectNodeOutput(engine.inputNode)
        engine.detach(current.sink)
        do {
            let sink = try makeSink(current.handler)
            capture = (current.handler, sink, current.clock)
            engine.prepare()
            try engine.start()
        } catch {
            onCaptureFailed?("The microphone stopped after an audio route change.")
        }
    }

    /// Builds the capture sink for the input's current hardware format. Caller holds the lock.
    private func makeSink(_ handler: @escaping CaptureHandler) throws -> AVAudioSinkNode {
        let input = engine.inputNode, source = input.outputFormat(forBus: 0)
        guard source.sampleRate > 0, source.channelCount > 0,
              let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: source, to: target) else {
            throw YoobError.unsupported("no microphone input is available")
        }
        converter.primeMethod = .none
        // A sink node reads the input in its realtime receiver; a tap can stop firing on the voice-processing graph.
        let sink = AVAudioSinkNode { @Sendable _, frames, buffers in
            guard let buffer = AVAudioPCMBuffer(pcmFormat: source, bufferListNoCopy: buffers, deallocator: nil) else { return noErr }
            buffer.frameLength = frames
            let capacity = AVAudioFrameCount(ceil(Double(frames) * 24000 / source.sampleRate)) + 32
            guard let converted = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return noErr }
            var supplied = false, error: NSError?
            let status = converter.convert(to: converted, error: &error) { _, state in
                if supplied { state.pointee = .noDataNow; return nil }
                supplied = true; state.pointee = .haveData; return buffer
            }
            guard status != .error, error == nil, converted.frameLength > 0, let samples = converted.int16ChannelData?[0] else { return noErr }
            var sum = 0.0
            for i in 0..<Int(converted.frameLength) { let value = Double(samples[i]) / 32768; sum += value * value }
            handler(Data(bytes: samples, count: Int(converted.frameLength) * 2), sqrt(sum / Double(converted.frameLength)))
            return noErr
        }
        engine.attach(sink)
        engine.connect(input, to: sink, format: source)
        return sink
    }

    /// Plays through the speaker and respects the ring/silent switch the way spoken audio should, unless the app has
    /// already chosen a category of its own.
    private static func activatePlaybackSession(force: Bool = false) throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        guard force || session.category == .soloAmbient else { return }
        try session.setCategory(.playback, mode: .spokenAudio)
        try session.setActive(true)
        #endif
    }

    func schedule(_ pcm: Data) {
        lock.lock()
        guard let format, pcm.count >= 2,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(pcm.count / 2)),
              let channel = buffer.floatChannelData?[0] else { lock.unlock(); return }
        let count = pcm.count / 2
        buffer.frameLength = AVAudioFrameCount(count)
        pcm.withUnsafeBytes { raw in
            for i in 0..<count { channel[i] = Float(Int16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))) / 32768 }
        }
        queued.append(Int64(count))
        let ticket = generation
        if !running {
            // The node's clock restarts at zero on play(); remember where in the utterance that is.
            // A drain's deferred stop may not have run yet; stop now (nothing is queued, so no callbacks fire).
            player.stop()
            runBase = finishedSamples; running = true
            player.play()
        }
        lock.unlock()
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            self?.finished(count: Int64(count), ticket: ticket)
        }
    }

    private func finished(count: Int64, ticket: Int) {
        lock.lock()
        guard ticket == generation, !queued.isEmpty else { lock.unlock(); return }
        queued.removeFirst()
        finishedSamples += count
        let drained = queued.isEmpty
        if drained { running = false }
        lock.unlock()
        guard drained else { return }
        // Never stop the node from its own completion callback; and skip the stop if new audio arrived meanwhile.
        control.async { [weak self] in
            guard let self else { return }
            self.lock.lock()
            let stillDrained = ticket == self.generation && self.queued.isEmpty && !self.running
            self.lock.unlock()
            if stillDrained { self.player.stop(); self.onDrained?() }
        }
    }

    var isIdle: Bool { lock.withLock { queued.isEmpty } }

    func stop() {
        lock.lock(); generation += 1; queued = []; running = false; lock.unlock()
        player.stop()
    }

    func shutdown() {
        stop()
        engine.stop()
    }
}
