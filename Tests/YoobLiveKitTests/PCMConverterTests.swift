import XCTest
import AVFoundation
@testable import YoobLiveKit

final class PCMConverterTests: XCTestCase {
    private func samples(_ data: Data) -> [Int16] {
        data.withUnsafeBytes { raw in (0..<data.count / 2).map { Int16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: Int16.self)) } }
    }

    func testInt16At24kPassesThrough() throws {
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 24_000, channels: 1, interleaved: false))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 240))
        buffer.frameLength = 240
        for i in 0..<240 { buffer.int16ChannelData![0][i] = Int16(i * 100 - 12_000) }
        let (pcm, level) = try XCTUnwrap(PCMConverter().convert(buffer))
        XCTAssertEqual(samples(pcm), (0..<240).map { Int16($0 * 100 - 12_000) })
        XCTAssertGreaterThan(level, 0.1)
    }

    func testStereoFloat48kIsDownmixedAndResampled() throws {
        let converter = PCMConverter()
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 2, interleaved: false))
        var output: [Int16] = []
        var levels: [Double] = []
        var phase = 0.0
        for _ in 0..<50 {   // half a second in 10 ms buffers, as WebRTC renders it
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 480))
            buffer.frameLength = 480
            for i in 0..<480 {
                let value = Float(0.5 * sin(phase))
                buffer.floatChannelData![0][i] = value
                buffer.floatChannelData![1][i] = value
                phase += 2 * .pi * 440 / 48_000
            }
            if let (pcm, level) = converter.convert(buffer) { output += samples(pcm); levels.append(level) }
        }
        // 12 000 samples in, give or take the resampler's filter delay.
        XCTAssertEqual(Double(output.count), 12_000, accuracy: 200)
        // A 0.5 sine in both channels stays a 0.5 sine (RMS ≈ 0.354) after the mono mix.
        XCTAssertEqual(levels.last ?? 0, 0.5 / 2.squareRoot(), accuracy: 0.03)
        XCTAssertEqual(Double(output.suffix(2_400).map { abs(Int($0)) }.max() ?? 0), 16_384, accuracy: 800)
    }

    func testInterleavedInt16StereoAndSilence() throws {
        let converter = PCMConverter()
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: 48_000, channels: 2, interleaved: true))
        var total = 0
        var lastLevel = 1.0
        for _ in 0..<20 {
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 480))
            buffer.frameLength = 480   // zero-filled: silence
            if let (pcm, level) = converter.convert(buffer) { total += pcm.count / 2; lastLevel = level }
        }
        XCTAssertEqual(Double(total), 4_800, accuracy: 200)
        XCTAssertEqual(lastLevel, 0)
    }

    func testEmptyBufferIsSkipped() throws {
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48_000, channels: 1, interleaved: false))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 480))
        XCTAssertNil(PCMConverter().convert(buffer))
    }
}
