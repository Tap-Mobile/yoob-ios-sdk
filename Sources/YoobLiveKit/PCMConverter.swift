import Foundation
@preconcurrency import AVFoundation

/// Converts whatever LiveKit renders (any rate, channel count, Int16/Int32/Float32, interleaved or not) to the
/// 24 kHz mono little-endian PCM16 the avatar takes. Keeps one resampler per input format so its filter state
/// carries across the 10 ms buffers WebRTC delivers. Thread-safe; called on the audio thread.
final class PCMConverter: @unchecked Sendable {
    static let outputRate = 24_000

    private let lock = NSLock()
    private var converter: AVAudioConverter?
    private let target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: Double(outputRate),
                                       channels: 1, interleaved: false)!

    /// The converted audio and its RMS level (0–1), or nil for an empty or unsupported buffer.
    func convert(_ buffer: AVAudioPCMBuffer) -> (pcm: Data, level: Double)? {
        lock.lock(); defer { lock.unlock() }
        let source = buffer.format
        guard buffer.frameLength > 0, source.sampleRate > 0, source.channelCount > 0 else { return nil }
        if converter?.inputFormat != source {
            guard let made = AVAudioConverter(from: source, to: target) else { return nil }
            made.primeMethod = .none
            made.downmix = true
            converter = made
        }
        guard let converter else { return nil }
        let capacity = AVAudioFrameCount(ceil(Double(buffer.frameLength) * target.sampleRate / source.sampleRate)) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return nil }
        var supplied = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, state in
            if supplied { state.pointee = .noDataNow; return nil }
            supplied = true
            state.pointee = .haveData
            return buffer
        }
        guard status != .error, error == nil, output.frameLength > 0, let samples = output.int16ChannelData?[0] else {
            return nil
        }
        let count = Int(output.frameLength)
        var sum = 0.0
        for i in 0..<count { let value = Double(samples[i]) / 32768; sum += value * value }
        // PCM16 is little-endian on every Apple platform, so the samples are copied as they are.
        return (Data(bytes: samples, count: count * 2), sqrt(sum / Double(count)))
    }
}
