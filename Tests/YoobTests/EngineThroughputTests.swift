import XCTest
@testable import Yoob
import YoobRealistic

/// Prints how fast each engine renders on this machine (frames per second of wall time). Needs YOOB_LOCAL_PACKS and
/// YOOB_SPEECH_PCM.
final class EngineThroughputTests: XCTestCase {
    func testThroughput() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let packs = env["YOOB_LOCAL_PACKS"], let speech = env["YOOB_SPEECH_PCM"] else { throw XCTSkip("no packs") }
        let pcm = try Data(contentsOf: URL(fileURLWithPath: speech))
        for character in ["luna-realistic", "luna-anime"] {
            let root = URL(fileURLWithPath: packs).appendingPathComponent(character)
            let manifest = try JSONDecoder().decode(CharacterManifest.self, from: Data(contentsOf: root.appendingPathComponent("character.json")))
            let clock = ContinuousClock()
            var start = clock.now
            let engine = try await FaceEngines.load(manifest, root: root)
            let loadTime = start.duration(to: clock.now)
            let resampler = try AvatarResampler()
            await engine.restart(hostFrame: 0)
            start = clock.now
            var samples: [Float] = []
            for offset in stride(from: 0, to: pcm.count, by: 9600) { samples += try resampler.convert(pcm.subdata(in: offset..<min(pcm.count, offset + 9600))) }
            samples += [Float](repeating: 0, count: engine.tailSamples)
            let feeding = Task { [samples] in try await engine.append(samples) }
            var frames = 0
            let expected = samples.count / 640 - 30
            var lastFrameAt = clock.now
            while frames < expected, lastFrameAt.duration(to: clock.now) < .seconds(3) {
                if try await engine.frame(frames) != nil { frames += 1; lastFrameAt = clock.now; if frames % 25 == 0 { print("\(character) frame \(frames) at \(start.duration(to: clock.now))") } } else { try await Task.sleep(for: .milliseconds(2)) }
            }
            try await feeding.value
            let elapsed = start.duration(to: lastFrameAt)
            let seconds = Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
            print("\(character): load \(loadTime), \(frames) frames in \(elapsed) = \(Double(frames) / seconds) fps")
        }
    }
}
