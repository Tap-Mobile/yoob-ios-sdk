import Foundation

/// One ordered producer awaits each append. Model calls are off the UI/audio threads.
public actor StreamingAvatar {
    private var models: AvatarModels
    private let pack: AvatarPack
    private var epoch = UUID()
    private var processing = false
    private var samples: [Float] = []
    private var sampleBase = 0
    private var rmsCursor = 0
    private var silenceStarts: [Int] = []
    /// Like `silenceStarts`, but below -30 dB: the call screen shows the idle face at roughly that output level, so the head
    /// walker treats it as the end of speech.
    private var quietStarts: [Int] = []
    private var nextFeature = 0
    private var nextFrame = 0
    private var features: [Int: [Float]] = [:]
    private var crops: [Int: Data] = [:]
    private var presentedFrame = -1
    /// Voice-first restarts: requests from an older segment get nothing, and `frameOffset` keeps the host
    /// video on the call's absolute frame so the head does not jump back to the start of its loop.
    private var segment = 0
    private var frameOffset = 0
    /// Host chosen for each rendered local frame, when the pack allows head motion beyond the calm stretch.
    private var hosts: [Int: Int] = [:]
    private var walker: HostWalker?
    public init(models: AvatarModels, pack: AvatarPack) { self.models = models; self.pack = pack }
    /// Continues on other models of the same pack (GPU to Neural Engine). Every encode and render is a pure function of its
    /// audio window and host frame, so the switch lands between two model calls with no state to carry: a prediction
    /// already running finishes on the old models, the next one uses the new ones.
    public func upgrade(to newer: AvatarModels) { models = newer }

    public func reset() {
        epoch = UUID(); processing = false; samples = []; sampleBase = 0; rmsCursor = 0
        silenceStarts = []; quietStarts = []; nextFeature = 0; nextFrame = 0; features = [:]; crops = [:]; presentedFrame = -1
        hosts = [:]; walker = nil
    }
    /// Starts a new segment whose frame 0 is absolute call frame `frameOffset`, as if the call began there.
    public func restart(segment: Int, frameOffset: Int) {
        reset(); self.segment = segment; self.frameOffset = max(0, frameOffset)
    }
    public func image(for frame: Int, segment expected: Int) throws -> AvatarImage? {
        guard expected == segment else { return nil }
        return try image(for: frame)
    }
    public func discard(before frame: Int, segment expected: Int) {
        guard expected == segment else { return }
        discard(before: frame)
    }
    public func append(samples incoming: [Float]) async throws -> Int {
        guard !processing, incoming.count <= 32000, incoming.allSatisfy(\.isFinite),
              sampleBase + samples.count + incoming.count <= 1200 * 16000 + 16000,
              samples.count + incoming.count <= 64000 else { throw AvatarError.invalidAudio }
        processing = true; let ticket = epoch
        defer { if epoch == ticket { processing = false } }
        samples.append(contentsOf: incoming)
        let available = sampleBase + samples.count
        while (rmsCursor + 1) * 640 <= available {
            let start = rmsCursor * 640 - sampleBase
            var power = 0.0
            for value in samples[start..<(start + 640)] { power += Double(value) * Double(value) }
            let db = 20 * log10(sqrt(power / 640 + 1e-12) + 1e-9)
            let previous = silenceStarts.last ?? -1
            silenceStarts.append(db < -40 ? (previous >= 0 ? previous : rmsCursor) : -1)
            let quiet = quietStarts.last ?? -1
            quietStarts.append(db < -30 ? (quiet >= 0 ? quiet : rmsCursor) : -1)
            rmsCursor += 1
        }
        while true {
            let low: Int, high: Int, first: Int, last: Int
            if nextFeature == 0 {
                guard available >= 8 * 640 + 80 else { break }
                low = 0; high = 8; first = 0; last = 8
            } else {
                guard available >= (nextFeature + 5) * 640 + 80 else { break }
                low = max(0, nextFeature - 16); high = nextFeature + 5; first = nextFeature; last = nextFeature + 1
            }
            let segment = Array(samples[(low * 640 - sampleBase)..<(high * 640 + 80 - sampleBase)])
            let encoded = try await models.encode(segment, frameCount: high - low)
            guard epoch == ticket, !Task.isCancelled else { throw CancellationError() }
            for frame in first..<last {
                let start = (frame - low) * 2048
                features[frame] = Array(encoded[start..<(start + 2048)])
            }
            nextFeature = last
            try await renderReady(ticket: ticket)
        }
        let retainFrom = max(0, nextFeature - 16) * 640
        if retainFrom > sampleBase { samples.removeFirst(retainFrom - sampleBase); sampleBase = retainFrom }
        return nextFrame - 1
    }
    private func renderReady(ticket: UUID) async throws {
        while nextFrame + 9 < nextFeature {
            guard crops.count < 60 else { throw AvatarError.invalidPack("renderer backpressure") }
            var window = [Float](repeating: 0, count: 40 * 1024)
            for relative in -10..<10 {
                if let feature = features[nextFrame + relative] {
                    let begin = (relative + 10) * 2048
                    window.replaceSubrange(begin..<(begin + 2048), with: feature)
                }
            }
            let weight = silenceWeight(nextFrame)
            if weight > 0 { for i in window.indices { window[i] = (1 - weight) * window[i] + weight * pack.closedAudio[i] } }
            let frame = nextFrame
            // The renderer's reference host must be the same absolute call frame the crop is composited into.
            var host: Int?
            if let calm = pack.calmHosts, calm.wideLast != nil {
                var walk = walker ?? HostWalker(window: calm, startHost: pack.hostIndex(for: frame + frameOffset))
                host = walk.next(speechAhead: speechAhead(from: frame))
                walker = walk; hosts[frame] = host
            }
            let crop = try await models.renderCrop(frame: frame + frameOffset, audio: window, host: host)
            guard epoch == ticket, !Task.isCancelled else { throw CancellationError() }
            crops[frame] = crop; nextFrame += 1
            features = features.filter { $0.key >= nextFrame - 10 }
        }
    }
    public func image(for frame: Int) throws -> AvatarImage? {
        guard frame >= 0, frame >= presentedFrame, let crop = crops[frame] else { return nil }
        let host = hosts[frame]
        let ahead = host == nil ? nil : (1...AvatarCompositor.prefetchFrames).compactMap { hosts[frame + $0] }
        let result = try AvatarCompositor.compose(pack: pack, frame: frame + frameOffset, cropBGR: crop, host: host,
                                                  prefetchHosts: ahead.flatMap { $0.isEmpty ? nil : $0 })
        presentedFrame = frame
        crops = crops.filter { $0.key >= frame }
        hosts = hosts.filter { $0.key >= frame }
        return result
    }
    public func discard(before frame: Int) { crops = crops.filter { $0.key >= frame } }
    /// Frames of speech known to follow `frame` before a quiet stretch of at least five frames (0 inside one).
    private func speechAhead(from frame: Int) -> Int {
        guard frame < quietStarts.count else { return 0 }
        var x = frame
        while x < quietStarts.count, x < frame + 120 {
            let start = quietStarts[x]
            if start >= 0, x - start >= 4 { return max(0, start - frame) }
            x += 1
        }
        return x - frame
    }
    private func silenceWeight(_ frame: Int) -> Float {
        guard frame < silenceStarts.count, silenceStarts[frame] >= 0 else { return 0 }
        let start = silenceStarts[frame]
        // At rendering time at least nine future feature frames are known. The five-frame
        // minimum and four-frame release can therefore be decided without unseen audio.
        guard start + 4 < silenceStarts.count, silenceStarts[start + 4] == start else { return 0 }
        var weight: Float = start == 0 ? 1 : min(1, Float(frame - start + 1) / 4)
        for distance in 1...4 where frame + distance < silenceStarts.count {
            if silenceStarts[frame + distance] < 0 { weight = min(weight, Float(distance) / 4); break }
        }
        return weight
    }
}
