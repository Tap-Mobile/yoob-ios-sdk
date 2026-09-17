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
            engine.stop()
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            self.format = format
            engine.prepare()
            do { try engine.start() } catch { throw YoobError.renderer("audio output: \(error.localizedDescription)") }
        }
        finishedSamples = 0; queued = []; runBase = 0; running = false
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
