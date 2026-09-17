import Foundation
import CoreML
import CoreGraphics

/// The existing Serve320 anime runtime, adapted to Live's continuous output clock.
/// Keeps the measured 25-frame geometry context and 8,000-sample encoder context.
/// No synthetic speech, per-packet end-of-utterance padding, or audio files.
public actor AnimeStreamingAvatar {
    private let bundle: Serve320Bundle
    private let geometry: MLModel
    private let renderer: MLModel
    private let encoder: Serve320Wav2Vec
    private let canvas = Serve320CanvasReader()
    private let temporal = Serve320TemporalBoostSession()
    private var pcm: [Float] = []
    private var baseFrame = 0
    private var nextFrame = 0
    private var windows: [(range: Range<Int>, pipeline: Serve320Pipeline)] = []
    /// Voice-first restarts: stale frame requests get nothing; the host motion loop keeps the absolute call frame.
    private var segment = 0
    private var frameOffset = 0
    private let compositor: Serve320MetalCompositor?
    private let payloadFrames: Int
    /// Frames in the host loop the compositor draws (`idle320_loop.mp4`, call frame mod this); the idle pose table must match.
    public nonisolated let idleFrameCount: Int
    /// DEBUG diagnostics: per-stage count / total / max milliseconds, to find where anime frame time goes.
    private var stages: [String: (count: Int, totalMS: Double, maxMS: Double)] = [:]

    private init(bundle: Serve320Bundle, geometry: MLModel, renderer: MLModel,
                 encoder: Serve320Wav2Vec, payloadFrames: Int, calmIdle: Bool) {
        self.bundle = bundle; self.geometry = geometry; self.renderer = renderer; self.encoder = encoder
        self.payloadFrames = payloadFrames; self.calmIdle = calmIdle
        idleFrameCount = bundle.nIdle
        compositor = Serve320MetalCompositor.createVerifiedForProduct(telemetry: { AvatarBenchmark.mark($0, $1) })
    }
    /// Anime idle frames are bundled (see `IdleSource`).
    /// Speech walks only the calm host frames the idle frames were rendered from. On when the host provides idle frames.
    private let calmIdle: Bool
    public static func load(payloadFrames: Int = 5, calmIdle: Bool = true) async throws -> AnimeStreamingAvatar {
        guard (5...30).contains(payloadFrames) else { throw Serve320Error.invalidOutput("anime payload size") }
        guard let root = Serve320Bundle.defaultRoot() else { throw Serve320Error.bundleMissing }
        let bundle = try Serve320Bundle(root: root)
        let geometry = try await ModelRegistry.shared.load("serve320.geometry_dynamic")
        // The actual native renderer is required, never silently substitute the H08 model.
        let renderer = try await ModelRegistry.shared.load("serve320.renderer_native224x160")
        let encoder = try await Serve320Wav2Vec.load()
        let stream = AnimeStreamingAvatar(bundle: bundle, geometry: geometry, renderer: renderer, encoder: encoder, payloadFrames: payloadFrames, calmIdle: calmIdle)
        try await stream.prepare()
        try await stream.warmUp()
        return stream
    }
    private func prepare() async throws { try await canvas.prepare() }
    /// Runs each call input shape (first window and steady window) once on silence during prewarm, so the first
    /// spoken window is not a cold Core ML specialization. The first real extraction measured 462 ms on iPhone Air.
    private func warmUp() async throws {
        for count in [payloadFrames + 25, payloadFrames + 50] {
            var start = ContinuousClock.now
            _ = try encoder.extract(pcm: [Float](repeating: 0, count: count * 640), nFrames: count, shift: 2,
                                    alignedStart: 0, alignedEnd: count * 640, globalFrameStart: 0)
            record("warmupEncoder", since: start)
            let input = try MLMultiArray(shape: [1, NSNumber(value: count), 3072], dataType: .float32)
            input.dataPointer.bindMemory(to: Float.self, capacity: count * 3072).initialize(repeating: 0, count: count * 3072)
            start = .now
            _ = try await geometry.prediction(from: MLDictionaryFeatureProvider(dictionary: ["feats": MLFeatureValue(multiArray: input)]))
            record("warmupGeometry", since: start)
        }
    }

    /// Starts a new segment whose frame 0 is absolute call frame `frameOffset`. The first window then uses the same
    /// left-edge context as the start of a call, and the recurrent temporal history is cleared.
    public func restart(segment: Int, frameOffset: Int) {
        pcm = []; baseFrame = 0; nextFrame = 0; windows = []
        self.segment = segment; self.frameOffset = max(0, frameOffset)
        temporal.reset(reason: "voice-first-restart")
    }
    public func image(for frame: Int, segment expected: Int) throws -> CGImage? {
        guard expected == segment else { return nil }
        return try image(for: frame)
    }

    public func append(samples: [Float]) async throws {
        try Task.checkCancellation()
        guard samples.allSatisfy(\.isFinite), samples.count <= 32_064 else { throw Serve320Error.invalidOutput("anime audio") }
        pcm.append(contentsOf: samples)
        // Six seconds is enough for one 30-frame payload, left/right context,
        // encoder support, and a maximum accepted packet. Never grow with call duration.
        guard pcm.count <= 96_000 else { throw Serve320Error.invalidOutput("anime processing fell behind") }
        while baseFrame * 640 + pcm.count >= (nextFrame + payloadFrames + 25) * 640 + 8_000 {
            try Task.checkCancellation()
            guard windows.count * payloadFrames < 180 else { throw Serve320Error.invalidOutput("anime playout fell behind") }
            let lo = max(0, nextFrame - 25), hi = nextFrame + payloadFrames + 25
            let localLo = lo - baseFrame, localHi = hi - baseFrame
            let count = hi - lo, keep = (nextFrame - lo)..<(nextFrame - lo + payloadFrames)
            let modelPCM = Array(pcm[(localLo * 640)..<(localHi * 640)])
            let rms = AnimeActivityGate.rms(modelPCM, frames: count)
            let activity = AnimeActivityGate.mask(rms)
            var stageStart = ContinuousClock.now
            let wav = try encoder.extract(pcm: pcm, nFrames: count, shift: 2,
                                          alignedStart: localLo * 640, alignedEnd: localHi * 640,
                                          globalFrameStart: localLo)
            record("encoderExtract", since: stageStart)
            let input = try MLMultiArray(shape: [1, NSNumber(value: count), 3072], dataType: .float32)
            let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: count * 3072)
            pointer.initialize(repeating: 0, count: count * 3072)
            for i in 0..<count {
                for j in 0..<1024 { pointer[i * 3072 + j] = Float(wav[i * 1024 + j]) }
                let phone = activity[i] ? 26 : 0
                pointer[i * 3072 + 1024 + phone] = 1
                pointer[i * 3072 + 2048 + phone] = 1
            }
            stageStart = .now
            let result = try await geometry.prediction(from: MLDictionaryFeatureProvider(dictionary: ["feats": MLFeatureValue(multiArray: input)]))
            record("geometryPrediction", since: stageStart)
            try Task.checkCancellation()
            guard let p6 = result.featureValue(for: "pred6")?.multiArrayValue,
                  let contact = result.featureValue(for: "contact_logit")?.multiArrayValue,
                  p6.count == count * 6, contact.count == count else { throw Serve320Error.invalidOutput("anime geometry") }
            let pred6 = keep.map { row in (0..<6).map { p6[row * 6 + $0].floatValue } }
            let contacts = keep.map { contact[$0].floatValue }
            stageStart = .now
            let pipeline = Serve320Pipeline(bundle: bundle, renderer: renderer, rendererProfile: .native224x160,
                                            temporalBoostSession: temporal, canvasReader: canvas)
            // Speech walks the host frames the anime idle frames were rendered from (IdleLoops/anime-frames: frames 0-5), at
            // the idle loop's 6 fps (25 / 4), so idle<->speech never changes the head pose.
            pipeline.hostMotionPolicy = calmIdle ? .calm(first: 0, count: 6, framesPerHost: 4) : .bundleLoop
            pipeline.hostFrameOffset = frameOffset + nextFrame
            pipeline.setExternalGeometry(pred6: pred6, contactLogit: contacts)
            if let boxes = bundle.idleRawBoxes() {
                pipeline.silenceEMA = Serve320SilenceEMA(rawBoxes: boxes, frameCount: payloadFrames, rms: Array(rms[keep]),
                    hardLock: Serve320SilenceEMA.hardLockDefault, energyReference: Serve320SilenceEMA.percentile90(rms))
            }
            pipeline.metalCompositor = compositor
            record("pipelineCreate", since: stageStart)
            windows.append((nextFrame..<(nextFrame + payloadFrames), pipeline))
            nextFrame += payloadFrames
            // Keep 25 geometry frames plus ceil(8000/640) encoder frames on the left.
            let retainFrom = max(0, nextFrame - 38)
            let remove = (retainFrom - baseFrame) * 640
            if remove > 0 { pcm.removeFirst(remove); baseFrame = retainFrom }
        }
    }
    public func image(for frame: Int) throws -> CGImage? {
        windows.removeAll { $0.range.upperBound <= frame }
        guard let window = windows.first(where: { $0.range.contains(frame) }) else { return nil }
        let begin = ContinuousClock.now
        let result = try autoreleasepool { try window.pipeline.compositedCGImage(frame: frame - window.range.lowerBound) }
        record(frame == window.range.lowerBound ? "compositeFirstInWindow" : "composite", since: begin)
        recordMS("rendererInference", result.out.rendererInferenceMs)
        return result.image
    }
    private func record(_ stage: String, since start: ContinuousClock.Instant) {
        let elapsed = start.duration(to: .now).components
        recordMS(stage, Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15)
    }
    private func recordMS(_ stage: String, _ ms: Double) {
        var entry = stages[stage] ?? (0, 0, 0); entry.count += 1; entry.totalMS += ms; entry.maxMS = max(entry.maxMS, ms); stages[stage] = entry
    }
    public func timingSnapshot() -> [String: [String: Double]] {
        stages.mapValues { ["count": Double($0.count), "meanMS": $0.totalMS / Double(max(1, $0.count)), "maxMS": $0.maxMS] }
    }
}

/// Activity-only branch of Serve320PhonemeFA for Realtime (which has no phoneme plan).
/// Matches the existing adaptive thresholds, one-frame pre-roll and one-frame gap fill.
enum AnimeActivityGate {
    static func rms(_ pcm: [Float], frames: Int) -> [Float] {
        (0..<frames).map { frame in
            let values = pcm[(frame * 640)..<min(pcm.count, (frame + 1) * 640)]
            return sqrt(values.reduce(Float(0)) { $0 + $1 * $1 } / Float(max(1, values.count)))
        }
    }
    static func mask(_ rms: [Float]) -> [Bool] {
        guard let peak = rms.max(), peak > 0 else { return Array(repeating: false, count: rms.count) }
        let sorted = rms.sorted()
        func percentile(_ fraction: Double) -> Float {
            sorted[Int((Double(sorted.count - 1) * fraction).rounded(.toNearestOrEven))]
        }
        let floor = min(percentile(0.05), percentile(0.10))
        let silence = max(5e-5, min(max(floor * 4, peak * 0.010), peak * 0.025))
        let threshold = max(0.010, peak * 0.040, silence * 1.5)
        var active = rms.map { $0 > threshold }
        for i in active.indices.filter({ active[$0] }) where i > 0 { active[i - 1] = true }
        if active.count > 2 {
            for i in 1..<(active.count - 1) where !active[i] && active[i - 1] && active[i + 1] { active[i] = true }
        }
        return active
    }
}

/// Existing renderer telemetry hook; normal app builds do not emit per-frame logs.
enum AvatarBenchmark {
    static let enabled = false
    static func mark(_ event: String, _ detail: String = "") {}
}
