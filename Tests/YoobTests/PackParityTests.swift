import XCTest
import CoreGraphics
@testable import YoobRealistic

/// The CDN pack keeps only the host frames the renderer can reach. This checks that the trimmed pack renders exactly
/// the frames the full pack renders. Set YOOB_FULL_PACK, YOOB_TRIMMED_PACK and YOOB_SPEECH_PCM (24 kHz mono PCM16).
final class PackParityTests: XCTestCase {
    func testTrimmedRealisticPackRendersIdenticalFrames() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let full = env["YOOB_FULL_PACK"], let trimmed = env["YOOB_TRIMMED_PACK"], let speech = env["YOOB_SPEECH_PCM"] else {
            throw XCTSkip("set YOOB_FULL_PACK, YOOB_TRIMMED_PACK and YOOB_SPEECH_PCM")
        }
        let pcm = try Data(contentsOf: URL(fileURLWithPath: speech))
        let a = try await render(pack: URL(fileURLWithPath: full), pcm: pcm)
        let b = try await render(pack: URL(fileURLWithPath: trimmed), pcm: pcm)
        XCTAssertGreaterThan(a.count, 50)
        XCTAssertEqual(a.count, b.count)
        for (index, (x, y)) in zip(a, b).enumerated() where x != y {
            XCTFail("frame \(index) differs"); break
        }
    }

    private func render(pack root: URL, pcm: Data) async throws -> [Data] {
        var pack = try AvatarPack(root: root)
        pack.calmHosts = .init(first: 1, count: 9, framesPerHost: 3, wideLast: 34)
        let models = try await AvatarModels.load(pack: pack, cpuOnly: true)
        let avatar = StreamingAvatar(models: models, pack: pack)
        let resampler = try AvatarResampler()
        var frames: [Data] = []
        var offset = 0
        var tail = false
        while true {
            let samples: [Float]
            if offset < pcm.count {
                let chunk = pcm.subdata(in: offset..<min(pcm.count, offset + 9600))
                offset += chunk.count
                samples = try resampler.convert(chunk)
            } else if !tail {
                tail = true; samples = [Float](repeating: 0, count: 16 * 640)
            } else { break }
            _ = try await avatar.append(samples: samples)
            while let image = try await avatar.image(for: frames.count) { frames.append(image.rgba) }
        }
        return frames
    }
}
