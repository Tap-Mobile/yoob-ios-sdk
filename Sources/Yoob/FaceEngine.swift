import Foundation
import CoreGraphics
import CoreML
import YoobRealistic
import YoobAnime

/// One character renderer: 16 kHz mono speech in, 25 fps frames out. Frame n shows the audio from n × 640 samples.
protocol FaceEngine: Sendable {
    func append(_ samples: [Float]) async throws
    /// The frame, once rendered; nil while it is still being prepared. Older frames are dropped.
    func frame(_ index: Int) async throws -> CGImage?
    /// Starts a new utterance. `hostFrame` keeps the head where the previous one left it.
    func restart(hostFrame: Int) async
    /// Silence to append after the last speech so the renderer's lookahead releases the final frames.
    var tailSamples: Int { get }
}

enum FaceEngines {
    static func load(_ manifest: CharacterManifest, root: URL) async throws -> FaceEngine {
        switch manifest.engine {
        case .realistic: try await RealisticEngine.load(manifest, root: root)
        case .anime: try await AnimeEngine.load(manifest, root: root)
        }
    }
}

final class RealisticEngine: FaceEngine {
    private let avatar: StreamingAvatar
    private let segment = SegmentCounter()
    private let pace = RenderPace(limit: 40)
    let tailSamples = 16 * 640

    private init(avatar: StreamingAvatar) { self.avatar = avatar }

    static func load(_ manifest: CharacterManifest, root: URL) async throws -> RealisticEngine {
        let pack: AvatarPack = try await Task.detached(priority: .userInitiated) {
            var pack = try AvatarPack(root: root)
            if let calm = manifest.calmHosts {
                pack.calmHosts = .init(first: calm.first, count: calm.count, framesPerHost: calm.framesPerHost, wideLast: calm.wideLast)
            }
            _ = try? pack.hostFrames.image(pack.hostIndex(for: 0), prefetch: [pack.hostIndex(for: 1), pack.hostIndex(for: 2)])
            return pack
        }.value
        // GPU: ready in well under a second once compiled. The Neural Engine's first specialization takes minutes.
        var models = try await AvatarModels.load(pack: pack, units: .cpuAndGPU)
        try await models.warmUp()
        // Some GPUs (the iOS Simulator's among them) return an empty picture from this renderer. Check one frame and
        // fall back to the CPU rather than show a black square.
        if try await rendersBlank(models, pack: pack) {
            models = try await AvatarModels.load(pack: pack, cpuOnly: true)
            if try await rendersBlank(models, pack: pack) { throw YoobError.renderer("the renderer produced an empty frame") }
        }
        return RealisticEngine(avatar: StreamingAvatar(models: models, pack: pack))
    }

    private static func rendersBlank(_ models: AvatarModels, pack: AvatarPack) async throws -> Bool {
        let crop = try await models.renderCrop(frame: 0, audio: pack.closedAudio)
        return !crop.contains { $0 > 8 }
    }

    func append(_ samples: [Float]) async throws {
        var offset = 0
        while offset < samples.count {
            // Audio can arrive much faster than real time; the renderer keeps at most `limit` frames ahead of the
            // frames being shown, so feed it a fifth of a second at a time and wait when it is far enough ahead.
            try await pace.waitForRoom()
            let slice = Array(samples[offset..<min(samples.count, offset + 3_200)])
            let rendered = try await avatar.append(samples: slice)
            await pace.rendered(through: rendered)
            offset += slice.count
        }
    }
    func frame(_ index: Int) async throws -> CGImage? {
        let current = await segment.value
        await pace.requested(index)
        await avatar.discard(before: max(0, index - 2), segment: current)
        return try await avatar.image(for: index, segment: current)?.cgImage()
    }
    func restart(hostFrame: Int) async {
        let next = await segment.next()
        await pace.reset()
        await avatar.restart(segment: next, frameOffset: hostFrame)
    }
}

final class AnimeEngine: FaceEngine {
    private let avatar: AnimeStreamingAvatar
    private let segment = SegmentCounter()
    /// One payload plus the geometry context and the encoder's right context.
    let tailSamples = (5 + 25) * 640 + 8_000

    private init(avatar: AnimeStreamingAvatar) { self.avatar = avatar }

    @MainActor static func load(_ manifest: CharacterManifest, root: URL) async throws -> AnimeEngine {
        YoobResources.root = root
        let avatar = try await AnimeStreamingAvatar.load(calmIdle: true)
        return AnimeEngine(avatar: avatar)
    }

    func append(_ samples: [Float]) async throws {
        var offset = 0
        while offset < samples.count {
            let slice = Array(samples[offset..<min(samples.count, offset + 32_000)])
            try await avatar.append(samples: slice)
            offset += slice.count
        }
    }
    func frame(_ index: Int) async throws -> CGImage? {
        try await avatar.image(for: index, segment: await segment.value)
    }
    func restart(hostFrame: Int) async {
        let next = await segment.next()
        await avatar.restart(segment: next, frameOffset: hostFrame)
    }
}

/// Holds the producer back while too many rendered frames wait to be shown.
actor RenderPace {
    private let limit: Int
    private var renderedThrough = -1
    private var requestedThrough = -1
    init(limit: Int) { self.limit = limit }
    func rendered(through frame: Int) { renderedThrough = max(renderedThrough, frame) }
    func requested(_ frame: Int) { requestedThrough = max(requestedThrough, frame) }
    func reset() { renderedThrough = -1; requestedThrough = -1 }
    func waitForRoom() async throws {
        while renderedThrough - requestedThrough >= limit {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

actor SegmentCounter {
    private(set) var value = 0
    func next() -> Int { value += 1; return value }
}
