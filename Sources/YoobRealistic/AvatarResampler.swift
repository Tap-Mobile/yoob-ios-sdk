import Foundation
@preconcurrency import AVFoundation

/// Stateful conversion to the models' 16 kHz preserves the stream's timing across packet boundaries.
public final class AvatarResampler {
    private let source: AVAudioFormat
    private let destination = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!
    private let converter: AVAudioConverter
    private let ratio: Double
    /// `sourceRate` is the PCM16 stream's rate: 24 kHz (OpenAI Realtime, the default), 16 kHz or 48 kHz.
    public init(sourceRate: Double = 24000) throws {
        guard [8000, 16000, 22050, 24000, 44100, 48000].contains(sourceRate),
              let source = AVAudioFormat(standardFormatWithSampleRate: sourceRate, channels: 1) else { throw AvatarError.invalidAudio }
        self.source = source; ratio = 16000 / sourceRate
        guard let converter = AVAudioConverter(from: source, to: destination) else { throw AvatarError.unavailable }
        converter.primeMethod = .none; self.converter = converter
    }
    public func convert(_ pcm: Data) throws -> [Float] {
        guard !pcm.isEmpty, pcm.count % 2 == 0, pcm.count <= Int(source.sampleRate) * 4,
              let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: AVAudioFrameCount(pcm.count / 2)),
              let pointer = input.floatChannelData?[0] else { throw AvatarError.invalidAudio }
        input.frameLength = AVAudioFrameCount(pcm.count / 2)
        pcm.withUnsafeBytes { bytes in
            for i in 0..<(pcm.count / 2) { pointer[i] = Float(Int16(littleEndian: bytes.loadUnaligned(fromByteOffset: i * 2, as: Int16.self))) / 32768 }
        }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * ratio)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: destination, frameCapacity: capacity) else { throw AvatarError.unavailable }
        var supplied = false, error: NSError?
        let result = converter.convert(to: output, error: &error) { _, status in
            if supplied { status.pointee = .noDataNow; return nil }
            supplied = true; status.pointee = .haveData; return input
        }
        guard result != .error, error == nil, let values = output.floatChannelData?[0] else { throw AvatarError.invalidAudio }
        return Array(UnsafeBufferPointer(start: values, count: Int(output.frameLength)))
    }
}
