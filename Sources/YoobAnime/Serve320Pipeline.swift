//
//  Serve320Pipeline.swift
//  The Serve320 serve lane: canned bundle features -> geometry CoreML (whole
//  500-frame clip in one call) -> per-frame landmarks + aligned ref + 27ch
//  renderer CoreML -> QA9 compositor onto the idle320 canvas -> display.
//
//  Mirrors arm_a/replay.py Replay.render (mode="tts", serve_stab, res=320 pack)
//  with the QA6/QA7/QA8/QA9 serve fixes ON and chin_cap_canvas="idle"
//  (bundle/meta.json serve_config). The QA13 output-EMA / QA15 hard-lock /
//  audio-gate serve_fixes_recommended items are intentionally NOT ported
//  (demo lane; listed in meta.json as recommendations only).
//
//  NOTE on dtypes: both .mlpackages declare FLOAT32 multiarray I/O at the
//  model boundary (fp16 is the internal compute/weight precision — verified
//  via coremltools spec: feats (1,500,3072) f32, x (1,27,320,320) f32).
//

import Foundation
import CoreML
import AVFoundation
import CoreVideo
import CoreGraphics
#if canImport(UIKit)
import UIKit
#endif

enum Serve320Error: Error {
    case modelMissing(String)
    case bundleMissing
    case invalidOutput(String)
    case canvasDecode(Int)
}

// MARK: - CoreML models (loading lives in ModelRegistry / models_manifest.json)

/// Spatial contract for the renderer graph. Native 224x160 is the only
/// production renderer. The full-320 graph remains available to explicit Debug
/// research builds, but Release neither selects nor bundles it.
struct Serve320RendererProfile: Equatable, Sendable {
    let name: String
    let registryID: String
    let width: Int
    let height: Int
    let x: Int
    let y: Int
    let edgeFeather: Int

    static let full320 = Serve320RendererProfile(
        name: "full320",
        registryID: "serve320.renderer",
        width: 320,
        height: 320,
        x: 0,
        y: 0,
        edgeFeather: 0)
    static let native224x160 = Serve320RendererProfile(
        name: "native224x160",
        registryID: "serve320.renderer_native224x160",
        width: 224,
        height: 160,
        x: 48,
        y: 96,
        edgeFeather: 6)

    static var requested: Serve320RendererProfile {
        #if DEBUG
        switch ProcessInfo.processInfo.environment[
            "AVATAR_SERVE320_RENDERER_PROFILE"]?.lowercased() {
        case "full320", "320", "quality", "control":
            return .full320
        default:
            return .native224x160
        }
        #else
        return .native224x160
        #endif
    }

    var isNativeROI: Bool { self != .full320 }

    /// Release fails closed when its required native ROI asset is missing or
    /// corrupt. Debug keeps an explicit full-320 research fallback so parity
    /// work remains possible without creating a production rollback path.
    static func loadRequestedRenderer() async throws
        -> (renderer: MLModel, profile: Serve320RendererProfile) {
        let requested = Self.requested
        #if DEBUG
        let selection = try await Serve320OptionalModelResolver.load(
            requestedID: requested.registryID,
            fallbackID: Self.full320.registryID,
            requestedIsOptional: requested.isNativeROI
        ) { registryID in
            try await ModelRegistry.shared.load(registryID)
        }
        let selectedProfile = selection.usedFallback ? Self.full320 : requested
        if selection.usedFallback {
            AvatarBenchmark.mark(
                "serve320_renderer_fallback",
                "requested=\(requested.name) selected=\(Self.full320.name) reason=native_model_unavailable"
            )
            NSLog(
                "[Serve320] optional renderer %@ unavailable; using %@",
                requested.name,
                Self.full320.name
            )
        }
        return (selection.value, selectedProfile)
        #else
        let renderer = try await ModelRegistry.shared.load(Self.native224x160.registryID)
        return (renderer, Self.native224x160)
        #endif
    }
}

final class Serve320Models {
    let geometry: MLModel
    let renderer: MLModel
    let rendererProfile: Serve320RendererProfile
    let placementSummary: String

    private init(geometry: MLModel, renderer: MLModel,
                 rendererProfile: Serve320RendererProfile,
                 placementSummary: String) {
        self.geometry = geometry
        self.renderer = renderer
        self.rendererProfile = rendererProfile
        self.placementSummary = placementSummary
    }

    static func load() async throws -> Serve320Models {
        // The enum-bucket ANE export duplicated the dynamic BiGRU at +9MB for
        // the canned lane only; both are the same wave3 weights. The canned
        // sample now shares the dynamic (CPU) model the live-text lane uses.
        let geo = try await ModelRegistry.shared.load("serve320.geometry_dynamic")
        let selection = try await Serve320RendererProfile.loadRequestedRenderer()
        return Serve320Models(
            geometry: geo,
            renderer: selection.renderer,
            rendererProfile: selection.profile,
            placementSummary: "units=cpu+ane renderer=\(selection.profile.name) in=feats,x out=pred6,contact_logit,y")
    }

    /// Single-model loader kept as a shim for legacy resource-name callers;
    /// compute-units defaults and quirks now live in models_manifest.json.
    @available(*, deprecated, message: "Use ModelRegistry.shared.load(<manifest id>) instead")
    static func loadNamed(_ resourceName: String,
                          computeUnits: MLComputeUnits? = nil) async throws -> MLModel {
        try await loadOne(resourceName: resourceName, computeUnits: computeUnits)
    }

    /// OTA pilot channel (AssetChannel.swift). With AVATAR_OTA=1 a pack in
    /// this channel can hotfix any Serve320 model by resource name — the
    /// renderer changed twice in one day, so it is the pilot use case.
    static let otaChannel = "renderer"

    private static func loadOne(resourceName: String,
                                computeUnits: MLComputeUnits? = nil) async throws -> MLModel {
        let config = MLModelConfiguration()
        // ANE-first (ModelRunner.swift:51); .all opt-in for GPU-path diagnostics.
        config.computeUnits = computeUnits ?? .cpuAndNeuralEngine
        config.allowLowPrecisionAccumulationOnGPU = true
        // AVATAR_OTA=1 (default off): a verified AssetChannel install wins over
        // Bundle.main; any override failure falls back to the bundled copy.
        if let override = await otaOverrideURL(resourceName: resourceName) {
            do {
                return try await MLModel.load(contentsOf: override, configuration: config)
            } catch {
                NSLog("[AssetChannel] OTA model %@ failed to load (%@); using bundled copy",
                      resourceName, String(describing: error))
            }
        }
        let url: URL
        if let mlmodelc = YoobResources.url(forResource: resourceName, withExtension: "mlmodelc") {
            url = mlmodelc
        } else if let mlpackage = YoobResources.url(forResource: resourceName,
                                                  withExtension: "mlpackage") {
            url = try await compileBundledPackageIfNeeded(mlpackage, cacheKey: resourceName)
        } else {
            throw Serve320Error.modelMissing(resourceName)
        }
        return try await MLModel.load(contentsOf: url, configuration: config)
    }

    /// Loadable model URL from the highest installed OTA version of
    /// `otaChannel`, or nil (flag off / nothing installed / bad payload —
    /// never throws, the bundled copy is always the fallback). Also kicks the
    /// once-per-process background refresh so a newer manifest version lands
    /// for the next load.
    private static func otaOverrideURL(resourceName: String) async -> URL? { nil }

    private static func compileBundledPackageIfNeeded(_ packageURL: URL,
                                                      cacheKey: String) async throws -> URL {
        let fm = FileManager.default
        let cacheRoot = try fm.url(for: .cachesDirectory, in: .userDomainMask,
                                   appropriateFor: nil, create: true)
            .appendingPathComponent("CompiledCoreML", isDirectory: true)
        try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cachedURL = cacheRoot.appendingPathComponent("\(cacheKey).mlmodelc",
                                                         isDirectory: true)
        if fm.fileExists(atPath: cachedURL.path) {
            return cachedURL
        }
        let compiledURL = try await MLModel.compileModel(at: packageURL)
        do {
            try fm.copyItem(at: compiledURL, to: cachedURL)
            return cachedURL
        } catch let error as CocoaError where error.code == .fileWriteFileExists {
            return cachedURL
        }
    }
}

// MARK: - Per-frame render output

struct Serve320FrameStageTimings {
    var canvasMs = 0.0
    var hostAndReferenceMs = 0.0
    var inputBuildMs = 0.0
    var inferenceMs = 0.0
    var outputAnalysisMs = 0.0
    var supportMs = 0.0
    var apertureSupportMs = 0.0
    var capMs = 0.0
    var skinMs = 0.0
    var targetMouthMs = 0.0
    var hostMouthMs = 0.0
    var supportCombineMs = 0.0
    var jawMs = 0.0
    var supportFinishMs = 0.0
    var jawSnapshotMs = 0.0
    var compositeMs = 0.0
    var metalUploadMs = 0.0
    var metalEncodeMs = 0.0
    var metalWaitMs = 0.0
    var metalGpuMs = 0.0
    var metalReadbackMs = 0.0
    var jawRestoreMs = 0.0
    var imageMs = 0.0
    var totalMs = 0.0

    mutating func add(_ other: Self) {
        canvasMs += other.canvasMs
        hostAndReferenceMs += other.hostAndReferenceMs
        inputBuildMs += other.inputBuildMs
        inferenceMs += other.inferenceMs
        outputAnalysisMs += other.outputAnalysisMs
        supportMs += other.supportMs
        apertureSupportMs += other.apertureSupportMs
        capMs += other.capMs
        skinMs += other.skinMs
        targetMouthMs += other.targetMouthMs
        hostMouthMs += other.hostMouthMs
        supportCombineMs += other.supportCombineMs
        jawMs += other.jawMs
        supportFinishMs += other.supportFinishMs
        jawSnapshotMs += other.jawSnapshotMs
        compositeMs += other.compositeMs
        metalUploadMs += other.metalUploadMs
        metalEncodeMs += other.metalEncodeMs
        metalWaitMs += other.metalWaitMs
        metalGpuMs += other.metalGpuMs
        metalReadbackMs += other.metalReadbackMs
        jawRestoreMs += other.jawRestoreMs
        imageMs += other.imageMs
        totalMs += other.totalMs
    }

    func divided(by count: Int) -> Self {
        let divisor = Double(max(count, 1))
        return Self(
            canvasMs: canvasMs / divisor,
            hostAndReferenceMs: hostAndReferenceMs / divisor,
            inputBuildMs: inputBuildMs / divisor,
            inferenceMs: inferenceMs / divisor,
            outputAnalysisMs: outputAnalysisMs / divisor,
            supportMs: supportMs / divisor,
            apertureSupportMs: apertureSupportMs / divisor,
            capMs: capMs / divisor,
            skinMs: skinMs / divisor,
            targetMouthMs: targetMouthMs / divisor,
            hostMouthMs: hostMouthMs / divisor,
            supportCombineMs: supportCombineMs / divisor,
            jawMs: jawMs / divisor,
            supportFinishMs: supportFinishMs / divisor,
            jawSnapshotMs: jawSnapshotMs / divisor,
            compositeMs: compositeMs / divisor,
            metalUploadMs: metalUploadMs / divisor,
            metalEncodeMs: metalEncodeMs / divisor,
            metalWaitMs: metalWaitMs / divisor,
            metalGpuMs: metalGpuMs / divisor,
            metalReadbackMs: metalReadbackMs / divisor,
            jawRestoreMs: jawRestoreMs / divisor,
            imageMs: imageMs / divisor,
            totalMs: totalMs / divisor
        )
    }
}

struct Serve320FrameOutput {
    let index: Int
    let rendererInferenceMs: Double // Core ML renderer prediction only
    let predCropBGR: [UInt8]      // 320x320x3 HWC raw renderer output (post QA8)
    let support: [Float]          // 320x320, post QA7/QA9 gates
    let box: (x0: Int, y0: Int, x1: Int, y1: Int)
    let landmarks: [SIMD2<Float>] // post-r2b decoded landmarks (crop coords)
    let aperture: Float
    let gate: Float               // r2b gate value (0 below threshold)
    let refRow: Int
    let targetHostCenterDx: Float // target landmark center - tracked idle-mouth center
    let hostMouthRecoveredPixels: Int // host non-skin pixels newly covered by the union gate
    let hostMouthUncoveredPixels: Int // tracked source-mouth pixels still outside effective support
    let mouthInkShiftX: Int       // bounded post-render correction in crop pixels
    let mouthInkRequestedShiftX: Float?
    let mouthInkBoundsShiftX: Float?
    let mouthInkCentroidShiftX: Float?
    let mouthInkResidualDx: Float?
    let rawMouthWidthToLandmarks: Float?
    let rawMouthHeightToLandmarks: Float?
    let rawMouthSharpness: Float?
    let rawMouthComponentCount: Int?
    let nativeROIHostFillApplied: Bool
    let temporalBoostHandled: Bool
    let temporalBoostApplied: Bool
    let jawProtectedSupportMax: Float
    let jawProtectedMask: [UInt8]
}

struct Serve320CompositeVisualMetrics {
    let idleMouthCenterX: Float?
    let outputMouthCenterX: Float?
    let idleMouthCentroidX: Float?
    let outputMouthCentroidX: Float?
    let outputMouthComponentCount: Int?
    let jawProtectedChangedPixels: Int
    let jawProtectedMaxChannelDelta: Int

    var relativeMouthCenterDx: Float? {
        guard let idleMouthCenterX, let outputMouthCenterX else { return nil }
        return outputMouthCenterX - idleMouthCenterX
    }

    var relativeMouthCentroidDx: Float? {
        guard let idleMouthCentroidX, let outputMouthCentroidX else { return nil }
        return outputMouthCentroidX - idleMouthCentroidX
    }
}

/// Result storage for the CPU support work that is overlapped with the ANE
/// renderer call. `DispatchWorkItem.wait()` is the synchronization boundary;
/// each box is written by exactly one work item and read only after it finishes.
private final class Serve320ParallelSupportBox: @unchecked Sendable {
    var apertureSupport: [Float] = []
    var skin: [Float] = []
    var apertureSupportMs = 0.0
    var skinMs = 0.0
}

/// Fixed renderer input provider whose `MLFeatureValue` is created once.
/// `rendererInput` is mutated in place between serial predictions, so rebuilding
/// an `MLDictionaryFeatureProvider` for every frame only adds allocation and
/// lookup overhead without changing the value Core ML sees.
private final class Serve320RendererInputProvider: MLFeatureProvider {
    let featureNames: Set<String> = ["x"]
    private let inputValue: MLFeatureValue

    init(input: MLMultiArray) {
        inputValue = MLFeatureValue(multiArray: input)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        featureName == "x" ? inputValue : nil
    }
}

/// Cross-queue result used by the prediction-first scheduling experiment.
/// The work-item completion is the synchronization boundary before any field
/// is read, and each render call owns a fresh box.
private final class Serve320RendererPredictionBox: @unchecked Sendable {
    var output: MLFeatureProvider?
    var error: Error?
    var elapsedMs = 0.0
}

/// Per-call result for idle-video decode. The work item is always waited before
/// any field is read, and each pipeline owns its own serial decode queue.
private final class Serve320CanvasFrameBox: @unchecked Sendable {
    var bytes: [UInt8]?
    var error: Error?
    var decodeMs = 0.0
}

// MARK: - Pipeline

final class Serve320Pipeline {
    private static func makeTemporalBoostSession() -> Serve320TemporalBoostSession {
        Serve320TemporalBoostSession { event, detail in
            AvatarBenchmark.mark(event, detail)
        }
    }

    let bundle: Serve320Bundle
    let renderer: MLModel
    let rendererProfile: Serve320RendererProfile
    private let geometryModel: MLModel?
    private let res = Serve320Bundle.res

    private(set) var pred6: [[Float]] = []        // (n,6) standardized geometry
    private(set) var contactLogit: [Float] = []   // (n,) raw logit

    /// Product playback may choose a stable host on lower-tier devices, while
    /// parity/eval callers keep the bundle loop by default. Set before the
    /// first composited frame and never change within a clip.
    var hostMotionPolicy: Serve320HostMotionPolicy = .bundleLoop
    /// Reply-global start for a chunk-local geometry window. Without this,
    /// every streamed window resets the head pose to idle row zero.
    var hostFrameOffset: Int = 0

    private var rendererInput: MLMultiArray?      // reusable profile-shaped f32
    private var rendererInputProvider: Serve320RendererInputProvider?
    // These allocation experiments are intentionally opt-in until repeated
    // physical-device full-path measurements clear their promotion gates.
    private let usesReusableRendererInputProvider =
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_REUSABLE_PROVIDER"] == "1"
    private let usesRendererOutputBacking =
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_OUTPUT_BACKING"] == "1"
    /// Core ML can write the fixed renderer output directly into this
    /// pipeline-owned buffer. Each pipeline renders serially, so one backing is
    /// safe to reuse and avoids an output allocation on every frame.
    private lazy var rendererPredictionOptions: MLPredictionOptions = {
        let options = MLPredictionOptions()
        if usesRendererOutputBacking,
           let output = try? MLMultiArray(
            shape: [
                1,
                3,
                NSNumber(value: rendererProfile.height),
                NSNumber(value: rendererProfile.width),
            ],
            dataType: .float32) {
            options.outputBackings = ["y": output]
        }
        return options
    }()
    private let canvas: Serve320CanvasReader
    private let canvasQueue = DispatchQueue(
        label: "com.yoob.avatardemo.serve320-canvas",
        qos: .userInitiated)
    private let rendererPredictionQueue = DispatchQueue(
        label: "com.yoob.avatardemo.serve320-renderer",
        qos: .userInitiated)
    private static let supportQueue = DispatchQueue(
        label: "com.yoob.avatardemo.serve320-support",
        qos: .userInitiated)
    private let usesPredictionFirstScheduling =
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_PREDICTION_FIRST"] == "1"
    /// Give Core ML a short submission head start before independent CPU mask
    /// work begins. Three cooled, adjacent physical-device A/B pairs improved
    /// both mean and p95 complete-frame time; an explicit zero retains the old
    /// scheduling path for rollback and future device-specific comparisons.
    private let supportHeadStartMicroseconds: Int = {
        let raw = ProcessInfo.processInfo.environment["AVATAR_SERVE320_ANE_HEADSTART_US"]
        return min(max(Int(raw ?? "") ?? 250, 0), 5_000)
    }()
    private let usesSerialSupportAfterPrediction =
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_ANE_FIRST_SERIAL"] == "1"
    /// Appearance-codebook retrieval hysteresis, ported from web 4349fa3.
    /// Selection is memoryless by construction, so the served exemplar changes
    /// on ~40% of consecutive frames. On the sealed 54-clip test the shipped
    /// web margin cut switching 36.6% (run length 2.47 -> 3.83 frames) with
    /// temporal_ratio 0.95146 and silence_flicker 0.96816, all 8 Eval v2 gates
    /// passing on both renderers and painted_lowpass significantly BETTER.
    /// An explicit zero restores exact memoryless selection for rollback.
    private let codebookHysteresisMargin: Float = {
        let raw = ProcessInfo.processInfo.environment["AVATAR_SERVE320_CODEBOOK_HYSTERESIS"]
        return min(max(Float(raw ?? "") ?? 0.25, 0), 4)
    }()
    /// Retrieval state. `previousCodebookFrame` exists because this pipeline —
    /// unlike the web worker — is NOT guaranteed to be driven sequentially:
    /// compositedCanvas(frame:) accepts an arbitrary index. Carrying a row
    /// across a discontinuity would make output depend on visit order, so the
    /// margin is applied only to a frame that directly follows its predecessor.
    private var previousCodebookRow = -1
    private var previousCodebookFrame = Int.min
    private(set) var lastFrameStageTimings: Serve320FrameStageTimings?
    private let mouthInkCorrectionLimit: Int = {
        let raw = ProcessInfo.processInfo.environment["AVATAR_MOUTH_INK_SHIFT_LIMIT"]
        return min(max(Int(raw ?? "") ?? 4, 0), 8)
    }()
    private let mouthInkCorrectionPolicy: String = {
        ProcessInfo.processInfo.environment["AVATAR_MOUTH_INK_SHIFT_POLICY"]?
            .lowercased() ?? "centroid"
    }()
    /// Research-only articulation recovery for the native 224x160 renderer.
    /// The graph is never selected by the default profile, and the boost itself
    /// remains opt-in until app-visible and physical-device gates pass.
    private let temporalBoostRequested =
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_TEMPORAL_BOOST"] == "1"
    private let temporalBoostSession: Serve320TemporalBoostSession
    private var temporalBoostLease: Serve320TemporalBoostLease

    init(bundle: Serve320Bundle, models: Serve320Models,
         temporalBoostSession requestedSession: Serve320TemporalBoostSession? = nil,
         canvasReader requestedCanvasReader: Serve320CanvasReader? = nil) {
        let session = requestedSession ?? Self.makeTemporalBoostSession()
        self.bundle = bundle
        self.renderer = models.renderer
        self.rendererProfile = models.rendererProfile
        self.geometryModel = models.geometry
        self.canvas = requestedCanvasReader ?? Serve320CanvasReader()
        self.temporalBoostSession = session
        self.temporalBoostLease = session.currentLease
        markCoreMLRuntimeConfiguration()
    }

    /// Live-text lane: renderer only; geometry comes precomputed per utterance
    /// (Serve320TextChain, bucketed) and is injected via setExternalGeometry.
    init(bundle: Serve320Bundle, renderer: MLModel,
         rendererProfile: Serve320RendererProfile = .requested,
         temporalBoostSession requestedSession: Serve320TemporalBoostSession? = nil,
         canvasReader requestedCanvasReader: Serve320CanvasReader? = nil) {
        let session = requestedSession ?? Self.makeTemporalBoostSession()
        self.bundle = bundle
        self.renderer = renderer
        self.rendererProfile = rendererProfile
        self.geometryModel = nil
        self.canvas = requestedCanvasReader ?? Serve320CanvasReader()
        self.temporalBoostSession = session
        self.temporalBoostLease = session.currentLease
        markCoreMLRuntimeConfiguration()
    }

    private func markCoreMLRuntimeConfiguration() {
        let schedule: String
        if usesSerialSupportAfterPrediction {
            schedule = "ane-first-serial"
        } else if supportHeadStartMicroseconds > 0 {
            schedule = "ane-headstart-\(supportHeadStartMicroseconds)us"
        } else if usesPredictionFirstScheduling {
            schedule = "queued-prediction-first"
        } else {
            schedule = "support-first"
        }
        AvatarBenchmark.mark(
            "serve320_coreml_runtime",
            "schedule=\(schedule) renderer=\(rendererProfile.name) output_backing=\(usesRendererOutputBacking ? 1 : 0) reusable_provider=\(usesReusableRendererInputProvider ? 1 : 0)")
    }

    /// Inject externally computed geometry (arbitrary n — live TTS path).
    func setExternalGeometry(pred6: [[Float]], contactLogit: [Float]) {
        self.pred6 = pred6
        self.contactLogit = contactLogit
        resetCodebookHysteresis()
    }

    /// New geometry means frame indices restart against unrelated content, so
    /// an exemplar held from the previous utterance must not leak across.
    private func resetCodebookHysteresis() {
        previousCodebookRow = -1
        previousCodebookFrame = Int.min
    }

    /// Explicit stream lifecycle hook for a new response, confirmed
    /// interruption/tombstone, seek, identity change, or teardown. Ordinary
    /// consecutive chunks must not call it; frame-ID continuity then carries
    /// the raw residual state across the chunk boundary.
    func resetTemporalBoost() {
        temporalBoostLease = temporalBoostSession.reset(reason: "pipeline_explicit_reset")
    }

    static func prepare() async throws -> Serve320Pipeline {
        guard let root = Serve320Bundle.defaultRoot() else {
            throw Serve320Error.bundleMissing
        }
        let bundle = try Serve320Bundle(root: root)
        let models = try await Serve320Models.load()
        let pipeline = Serve320Pipeline(bundle: bundle, models: models)
        try await pipeline.prepareCanvas()
        try pipeline.runGeometry()
        return pipeline
    }

    /// Async one-time canvas reader init (AVAssetTrack load is async on iOS 17+).
    func prepareCanvas() async throws {
        try await canvas.prepare()
    }

    /// Whole-clip geometry in one CoreML call — the exported chain was traced at
    /// N=500 with NO tail-pad (meta.json serve_config.geometry_input).
    func runGeometry() throws {
        let n = Serve320Bundle.nFeat
        let input = try MLMultiArray(shape: [1, NSNumber(value: n), 3072], dataType: .float32)
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: n * 3072)
        for i in 0..<n {
            bundle.featureRow(i, into: ptr + i * 3072)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "feats": MLFeatureValue(multiArray: input),
        ])
        guard let geometryModel else {
            throw Serve320Error.invalidOutput("geometry model (use setExternalGeometry for the live lane)")
        }
        let out = try geometryModel.prediction(from: provider)
        guard let p6 = out.featureValue(for: "pred6")?.multiArrayValue,
              let cl = out.featureValue(for: "contact_logit")?.multiArrayValue,
              p6.count == n * 6, cl.count == n else {
            throw Serve320Error.invalidOutput("geometry outputs")
        }
        let p6Ptr = p6.dataPointer.bindMemory(to: Float.self, capacity: p6.count)
        let clPtr = cl.dataPointer.bindMemory(to: Float.self, capacity: cl.count)
        pred6 = (0..<n).map { i in (0..<6).map { p6Ptr[i * 6 + $0] } }
        contactLogit = (0..<n).map { clPtr[$0] }
        resetCodebookHysteresis()
    }

    /// One serve frame: build the 27ch renderer input, run the renderer, then the
    /// QA7/8/9 support gates. Mirrors replay.py:484-576 (per-frame loop body).
    func renderPredCrop(frame i: Int) throws -> Serve320FrameOutput {
        try renderPredCrop(frame: i, hostFrame: hostFrameIndex(for: i))
    }

    private func renderPredCrop(frame i: Int, hostFrame h: Int) throws -> Serve320FrameOutput {
        let profiling = AvatarBenchmark.enabled
        var stageMark = profiling ? DispatchTime.now().uptimeNanoseconds : 0
        var stageTimings = profiling
            ? (lastFrameStageTimings ?? Serve320FrameStageTimings()) : nil
        guard i >= 0, i < pred6.count else {
            throw Serve320Error.invalidOutput("frame index \(i)")
        }
        let pca = bundle.pca
        let geom = Serve320Math.reconstructGeometry(pred6[i], pca: pca)
        let ap = Serve320Math.apertureFromGeometry(geom)   // replay.py:486

        // host crop: planar BGR u8 -> HWC (replay.py:488-492 crop at the stab box ==
        // idle_crops320 row; crop==composite space)
        let hostUnwarped = Self.planarBGRToHWC(bundle.idleCrop(h), res: res)
        let contour = bundle.idleContour(h)
        // QA6 R2 chin warp of the host BEFORE the renderer input (replay.py:500-501;
        // render_utils.chin_warp:287-325; serve calls with default 16/28 at 320,
        // mouth_floor = int(0.62*320) = 198 — meta.json qa9_compositor.chin_warp)
        let hostWarped = Self.chinWarp(hostUnwarped, contourU8: contour, aperture: ap,
                                       mouthFloor: Int(0.62 * Float(res)),
                                       maxWarpPx: 16, rampPx: 28)

        // landmarks: decode pred6 with the frame anchor (idle rows == canon anchor),
        // then R2b closure gate blend (replay.py:513-516)
        let anchor = bundle.anchor(h)
        var tgtLm = Serve320Math.decodePointsPerframe(pred6[i], anchor: anchor, pca: pca)
        let gate = Serve320Math.r2bGate(logit: contactLogit[i],
                                        threshold: Serve320Bundle.r2bThreshold)
        tgtLm = Serve320Math.r2bBlend(tgtLm, gate: gate,
                                      tmplYoff: Serve320Bundle.r2bTemplateYoff,
                                      strength: Serve320Bundle.r2bStrength)
        let hostCenter = bundle.idleMouthCenter320(h)
        let predictedCenterX = tgtLm.reduce(Float(0)) { $0 + $1.x } / Float(tgtLm.count)
        let centerShiftX = hostCenter.x - predictedCenterX
        for j in tgtLm.indices { tgtLm[j].x += centerShiftX }

        // These values and the two expensive ownership masks depend only on
        // geometry + the host frame, never on renderer output. Compute them on
        // the CPU while Core ML is executing on the ANE instead of serializing
        // another ~4 ms after prediction.
        var lowerLipY: Float = 0
        var mouthCx: Float = 0
        for p in tgtLm {
            lowerLipY = max(lowerLipY, p.y)
            mouthCx += p.x
        }
        mouthCx /= Float(tgtLm.count)
        let capBase = max(lowerLipY, bundle.hostLipY320(h))

        // aligned appearance-codebook ref (replay.py:457, 519-523), with the
        // web-shipped retrieval hysteresis. The previous row is offered only
        // when this frame directly follows the one that produced it, so a
        // random-access render is bit-identical to memoryless selection.
        let previousRow = (i == previousCodebookFrame + 1) ? previousCodebookRow : -1
        let refRow = Serve320Math.appearanceCodebookRef(query: pred6[i], aperture: ap,
                                                        codebook: bundle.codebook,
                                                        refGeom6: bundle.refGeom6Flat,
                                                        previousRow: previousRow,
                                                        margin: codebookHysteresisMargin)
        previousCodebookRow = refRow
        previousCodebookFrame = i
        let refLm = Serve320Math.decodePointsPerframe(bundle.refGeom(refRow),
                                                      anchor: bundle.refAnchor(refRow),
                                                      pca: pca)
        let refRGB01 = Self.planarBGRToRGB01(bundle.refCrop(refRow), res: res)
        let refAligned = Serve320Math.alignRef(refRGB01: refRGB01, res: res,
                                               refPts: refLm, tgtPts: tgtLm)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.hostAndReferenceMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }

        // ---- 27ch input (replay.py:524; train_renderer_256.py:146 Data256.inp — SAME
        // order train==serve):
        //   ch 0-2  : masked host RGB01   (host * (1 - hole))
        //   ch 3-5  : aligned-ref RGB01   (align_ref output, UNMASKED by design)
        //   ch 6    : input hole          (input_mask320, 1 = hole)
        //   ch 7-26 : 20 landmark gaussian heatmaps, sigma 4.0 (models.py:115-124)
        let input = try rendererInputArray()
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let plane = res * res
        let rendererWidth = rendererProfile.width
        let rendererHeight = rendererProfile.height
        let rendererPlane = rendererWidth * rendererHeight
        let hole = bundle.hole
        // Rows write disjoint renderer-plane ranges; per-pixel math unchanged.
        hostWarped.withUnsafeBufferPointer { hostBuf in
            refAligned.withUnsafeBufferPointer { refBuf in
                hole.withUnsafeBufferPointer { holeBuf in
                    let hw = hostBuf.baseAddress!
                    let ra = refBuf.baseAddress!
                    let ho = holeBuf.baseAddress!
                    Serve320ImageOps.fanOutLines(0, rendererHeight - 1) { cy0, cy1 in
                        for localY in cy0...cy1 {
                            let globalY = rendererProfile.y + localY
                            for localX in 0..<rendererWidth {
                                let globalX = rendererProfile.x + localX
                                let globalPixel = globalY * res + globalX
                                let rendererPixel = localY * rendererWidth + localX
                                let keep = 1 - ho[globalPixel]
                                let b = Float(hw[globalPixel * 3]) / 255
                                let g = Float(hw[globalPixel * 3 + 1]) / 255
                                let r = Float(hw[globalPixel * 3 + 2]) / 255
                                ptr[0 * rendererPlane + rendererPixel] = r * keep
                                ptr[1 * rendererPlane + rendererPixel] = g * keep
                                ptr[2 * rendererPlane + rendererPixel] = b * keep
                                ptr[3 * rendererPlane + rendererPixel] = ra[0 * plane + globalPixel]
                                ptr[4 * rendererPlane + rendererPixel] = ra[1 * plane + globalPixel]
                                ptr[5 * rendererPlane + rendererPixel] = ra[2 * plane + globalPixel]
                                ptr[6 * rendererPlane + rendererPixel] = ho[globalPixel]
                            }
                        }
                    }
                }
            }
        }
        let rendererPoints = tgtLm.map {
            SIMD2<Float>(
                $0.x - Float(rendererProfile.x),
                $0.y - Float(rendererProfile.y))
        }
        Serve320Math.landmarkHeatmaps(
            points: rendererPoints,
            width: rendererWidth,
            height: rendererHeight,
            sigma: 4.0,
            dst: ptr,
            channelOffset: 7)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.inputBuildMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }

        let supportBox = Serve320ParallelSupportBox()
        let supportWork = DispatchWorkItem {
            let apertureStart = profiling ? DispatchTime.now().uptimeNanoseconds : 0
            supportBox.apertureSupport = Serve320Compositor.apertureGatedSupport(
                support: self.bundle.support, contourU8: contour, aperture: ap,
                hole: self.bundle.hole, maxBandPx: 70, featherPx: 2)
            if profiling {
                let now = DispatchTime.now().uptimeNanoseconds
                supportBox.apertureSupportMs = Self.milliseconds(
                    from: apertureStart, to: now)
                let skinStart = now
                supportBox.skin = Serve320Compositor.hostSkinMask(
                    host: hostUnwarped, cx: mouthCx, capY: capBase,
                    k: 14, tol: 60, closePx: 5)
                supportBox.skinMs = Self.milliseconds(
                    from: skinStart, to: DispatchTime.now().uptimeNanoseconds)
            } else {
                supportBox.skin = Serve320Compositor.hostSkinMask(
                    host: hostUnwarped, cx: mouthCx, capY: capBase,
                    k: 14, tol: 60, closePx: 5)
            }
        }
        let provider: MLFeatureProvider
        if !usesReusableRendererInputProvider {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                "x": MLFeatureValue(multiArray: input),
            ])
        } else if let rendererInputProvider {
            provider = rendererInputProvider
        } else {
            let reusable = Serve320RendererInputProvider(input: input)
            rendererInputProvider = reusable
            provider = reusable
        }

        let out: MLFeatureProvider
        let rendererInferenceMs: Double
        if usesSerialSupportAfterPrediction {
            let inferenceStart = DispatchTime.now().uptimeNanoseconds
            out = try renderer.prediction(from: provider, options: rendererPredictionOptions)
            rendererInferenceMs = Double(
                DispatchTime.now().uptimeNanoseconds - inferenceStart) / 1_000_000.0
            supportWork.perform()
        } else if supportHeadStartMicroseconds > 0 {
            // Keep prediction on the current render executor, but defer the
            // independent CPU work just long enough for Core ML to submit the
            // request. The masks still finish well inside a normal ANE call.
            Self.supportQueue.asyncAfter(
                deadline: .now() + .microseconds(supportHeadStartMicroseconds),
                execute: supportWork)
            let inferenceStart = DispatchTime.now().uptimeNanoseconds
            out = try renderer.prediction(from: provider, options: rendererPredictionOptions)
            rendererInferenceMs = Double(
                DispatchTime.now().uptimeNanoseconds - inferenceStart) / 1_000_000.0
        } else if usesPredictionFirstScheduling {
            // Give Core ML the first scheduling opportunity, then do the
            // independent ownership-mask work on this thread while the ANE is
            // busy. The production ordering does the inverse. This switch is
            // deliberately opt-in until a physical 500-frame A/B proves that
            // the order improves the complete path, not just the model call.
            let predictionBox = Serve320RendererPredictionBox()
            let predictionWork = DispatchWorkItem {
                let start = DispatchTime.now().uptimeNanoseconds
                do {
                    predictionBox.output = try self.renderer.prediction(
                        from: provider,
                        options: self.rendererPredictionOptions)
                } catch {
                    predictionBox.error = error
                }
                predictionBox.elapsedMs = Double(
                    DispatchTime.now().uptimeNanoseconds - start) / 1_000_000.0
            }
            rendererPredictionQueue.async(execute: predictionWork)
            supportWork.perform()
            predictionWork.wait()
            if let error = predictionBox.error { throw error }
            guard let predictionOutput = predictionBox.output else {
                throw Serve320Error.invalidOutput("renderer prediction")
            }
            out = predictionOutput
            rendererInferenceMs = predictionBox.elapsedMs
        } else {
            Self.supportQueue.async(execute: supportWork)
            let inferenceStart = DispatchTime.now().uptimeNanoseconds
            out = try renderer.prediction(from: provider, options: rendererPredictionOptions)
            rendererInferenceMs = Double(
                DispatchTime.now().uptimeNanoseconds - inferenceStart) / 1_000_000.0
        }
        if profiling {
            stageTimings?.inferenceMs = rendererInferenceMs
            stageMark = DispatchTime.now().uptimeNanoseconds
        }
        guard let y = out.featureValue(for: "y")?.multiArrayValue,
              y.count == 3 * rendererPlane else {
            throw Serve320Error.invalidOutput("renderer output")
        }
        // rgb01 = (tanh+1)/2 -> BGR u8 (render_utils.rgb01_to_bgr_u8:239-242:
        // clamp, *255+0.5, astype(u8) — truncation == round-half-up)
        let yPtr = y.dataPointer.bindMemory(to: Float.self, capacity: y.count)
        var predCrop = hostWarped
        predCrop.withUnsafeMutableBufferPointer { crop in
            let cp = crop.baseAddress!
            Serve320ImageOps.fanOutLines(0, rendererHeight - 1) { cy0, cy1 in
                for localY in cy0...cy1 {
                    let globalY = rendererProfile.y + localY
                    for localX in 0..<rendererWidth {
                        let globalX = rendererProfile.x + localX
                        let rendererPixel = localY * rendererWidth + localX
                        let globalPixel = globalY * res + globalX
                        let rr = min(max((yPtr[0 * rendererPlane + rendererPixel] + 1) / 2, 0), 1)
                        let gg = min(max((yPtr[1 * rendererPlane + rendererPixel] + 1) / 2, 0), 1)
                        let bb = min(max((yPtr[2 * rendererPlane + rendererPixel] + 1) / 2, 0), 1)
                        cp[globalPixel * 3] = UInt8(min(bb * 255 + 0.5, 255))
                        cp[globalPixel * 3 + 1] = UInt8(min(gg * 255 + 0.5, 255))
                        cp[globalPixel * 3 + 2] = UInt8(min(rr * 255 + 0.5, 255))
                    }
                }
            }
        }

        // The renderer may place visible lip pigment a few pixels away from its
        // already-centered landmark heatmaps. Correct the pigment-mass residual
        // rather than the outer bounds: asymmetric open-mouth shapes made the
        // old bounds correction push the visible mouth in the wrong direction.
        // All host-space ownership masks and the stab box stay fixed so the
        // old-mouth and jaw protections remain intact.
        let hostInkMetrics = Serve320Compositor.mouthInkMetrics(
            imageBGR: hostUnwarped, width: res, height: res,
            expectedX: hostCenter.x, expectedY: hostCenter.y)
        let predInkMetrics = Serve320Compositor.mouthInkMetrics(
            imageBGR: predCrop, width: res, height: res,
            expectedX: hostCenter.x, expectedY: hostCenter.y)
        let rawMouthSharpness = predInkMetrics.flatMap {
            Serve320Compositor.mouthSharpness(
                imageBGR: predCrop, width: res, height: res, metrics: $0)
        }
        let mouthInkBoundsShift = hostInkMetrics.flatMap { host in
            predInkMetrics.map { host.boundsCenterX - $0.boundsCenterX }
        }
        let mouthInkCentroidShift = hostInkMetrics.flatMap { host in
            predInkMetrics.map { host.centroidX - $0.centroidX }
        }
        let requestedMouthInkShift = mouthInkBoundsShift.flatMap { boundsDelta in
            mouthInkCentroidShift.map { centroidDelta in
                if rendererProfile.isNativeROI { return Float(0) }
                switch mouthInkCorrectionPolicy {
                case "off", "none": return Float(0)
                case "bounds": return boundsDelta
                case "centroid": return centroidDelta
                default:
                    return Serve320Compositor.guardedMouthInkShift(
                        boundsDelta: boundsDelta,
                        centroidDelta: centroidDelta)
                }
            }
        }
        let mouthInkShiftX: Int
        if let requestedMouthInkShift {
            mouthInkShiftX = Serve320Compositor.boundedMouthInkShift(
                requested: requestedMouthInkShift,
                limit: mouthInkCorrectionLimit)
            predCrop = Serve320Compositor.translateBGRHorizontally(
                predCrop, width: res, height: res, dx: mouthInkShiftX)
        } else {
            mouthInkShiftX = 0
        }
        let mouthInkResidualDx = requestedMouthInkShift.map {
            $0 - Float(mouthInkShiftX)
        }
        let landmarkMinX = tgtLm.map(\.x).min() ?? 0
        let landmarkMaxX = tgtLm.map(\.x).max() ?? 0
        let landmarkMinY = tgtLm.map(\.y).min() ?? 0
        let landmarkMaxY = tgtLm.map(\.y).max() ?? 0
        let landmarkWidth = max(landmarkMaxX - landmarkMinX, 1)
        let landmarkHeight = max(landmarkMaxY - landmarkMinY, 1)
        let rawMouthWidthToLandmarks = predInkMetrics.map {
            Float($0.width) / landmarkWidth
        }
        let rawMouthHeightToLandmarks = predInkMetrics.map {
            Float($0.height) / landmarkHeight
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.outputAnalysisMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        supportWork.wait()
        // ---- QA9 support stack (replay.py:543-569; meta.json qa9_compositor)
        var support = supportBox.apertureSupport
        if profiling {
            stageTimings?.apertureSupportMs = supportBox.apertureSupportMs
        }
        // QA8: cap clears BOTH mouths (replay.py:552-555; idle_host_lip_y320)
        let capStart = profiling ? DispatchTime.now().uptimeNanoseconds : 0
        Serve320Compositor.capDcCorrect(pred: &predCrop, host: hostUnwarped,
                                        lipY: lowerLipY, capY: capBase + 4.0,
                                        mask: support, cx: mouthCx, k: 14)  // replay.py:559-561
        let capW = Serve320Compositor.lipCapWeight(capBase: capBase,
                                                   marginPx: 4.0, featherPx: 14.0)
        for yy in 0..<res {
            let w = capW[yy]
            guard w < 1 else { continue }
            let row = yy * res
            for xx in 0..<res {
                support[row + xx] *= w                   // replay.py:562
            }
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.capMs = Self.milliseconds(from: capStart, to: now)
            stageMark = now
        }
        // QA9 skinmask: blend may touch host-skin only, mouth polygon exempt
        // (replay.py:563-569; host_skin_mask render_utils.py:404-424)
        let skin = supportBox.skin
        if profiling {
            stageTimings?.skinMs = supportBox.skinMs
        }
        let lipInt = tgtLm.map { SIMD2<Int>(Int($0.x), Int($0.y)) }   // astype(int32)
        let hull = Serve320ImageOps.convexHull(lipInt)
        var mouth = Serve320ImageOps.fillConvexPoly(hull, w: res, h: res)
        // ±10px past the outer lip (was 13 → ±6px): under anchor lag at host
        // head-bobs the HOST lip can sit ~10px outside the predicted hull, and
        // the skinmask treats lip pigment as protected non-skin — the surviving
        // host lip beside the rendered one is the "double lip" artifact.
        let mouthRadius = 21 / 2
        let mouthWindow = hull.isEmpty ? nil : Serve320ImageOps.RectWindow(
            x0: max(0, (hull.map(\.x).min() ?? 0) - mouthRadius),
            y0: max(0, (hull.map(\.y).min() ?? 0) - mouthRadius),
            x1: min(res - 1, (hull.map(\.x).max() ?? res - 1) + mouthRadius),
            y1: min(res - 1, (hull.map(\.y).max() ?? res - 1) + mouthRadius))
        mouth = Serve320ImageOps.dilateRect(
            mouth, w: res, h: res, k: 21, window: mouthWindow)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.targetMouthMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        let hostMouth = Serve320Compositor.hostMouthResidualMask(
            skin: skin, hole: bundle.hole, contourU8: contour, center: hostCenter,
            lowerLipY: bundle.hostLipY320(h), anchorWidth: anchor.width)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.hostMouthMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        // Full-plane combine: chunks accumulate integer partials (order-free)
        // and write disjoint support ranges.
        var recoveredPixels = 0
        let recoveredLock = NSLock()
        support.withUnsafeMutableBufferPointer { sup in
            skin.withUnsafeBufferPointer { skinBuf in
                mouth.withUnsafeBufferPointer { mouthBuf in
                    hostMouth.withUnsafeBufferPointer { hostMouthBuf in
                        let s = sup.baseAddress!
                        let sk = skinBuf.baseAddress!
                        let mo = mouthBuf.baseAddress!
                        let hm = hostMouthBuf.baseAddress!
                        Serve320ImageOps.fanOutLines(0, res - 1) { cy0, cy1 in
                            var recovered = 0
                            for px in (cy0 * res)..<((cy1 + 1) * res) {
                                let targetGate = max(sk[px], mo[px])
                                if s[px] > 0.05, targetGate < 0.5, hm[px] >= 0.5 {
                                    recovered += 1
                                }
                                s[px] *= max(targetGate, hm[px])
                            }
                            recoveredLock.lock()
                            recoveredPixels += recovered
                            recoveredLock.unlock()
                        }
                    }
                }
            }
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.supportCombineMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }

        // Keep a five-pixel ring of the tracked host silhouette untouched. The
        // QA9 skin close can bridge over the thin dark jaw stroke and otherwise
        // lets the soft renderer repaint it, producing a broken chin contour.
        var hostFace = [Float](repeating: 0, count: plane)
        var faceX0 = res, faceX1 = -1, faceY0 = res, faceY1 = -1
        let faceBoxes = Serve320ParallelBoxes(rows: res)
        hostFace.withUnsafeMutableBufferPointer { face in
            let fp = face.baseAddress!
            Serve320ImageOps.fanOutLines(0, res - 1) { cy0, cy1 in
                var lx0 = res, lx1 = -1, ly0 = res, ly1 = -1
                for y in cy0...cy1 {
                    let row = y * res
                    for x in 0..<res where contour[row + x] >= 128 {
                        fp[row + x] = 1
                        lx0 = min(lx0, x); lx1 = max(lx1, x)
                        ly0 = min(ly0, y); ly1 = max(ly1, y)
                    }
                }
                faceBoxes.merge(x0: lx0, x1: lx1, y0: ly0, y1: ly1)
            }
        }
        (faceX0, faceX1, faceY0, faceY1) = faceBoxes.box(defaultX: res, defaultY: res)
        let faceWindow = faceX1 >= faceX0 && faceY1 >= faceY0
            ? Serve320ImageOps.RectWindow(
                x0: faceX0, y0: faceY0, x1: faceX1, y1: faceY1)
            : nil
        let jawInterior = Serve320ImageOps.erodeRectBinary(
            hostFace, w: res, h: res, k: 11, window: faceWindow)
        support.withUnsafeMutableBufferPointer { sup in
            jawInterior.withUnsafeBufferPointer { jaw in
                let s = sup.baseAddress!
                let j = jaw.baseAddress!
                Serve320ImageOps.fanOutLines(0, res - 1) { cy0, cy1 in
                    for px in (cy0 * res)..<((cy1 + 1) * res) { s[px] *= j[px] }
                }
            }
        }
        // Preserve only the ROI-sized generated support before the boundary
        // feather. Keeping a second full 320² Float array here would add ~400 KB
        // of copy traffic per frame on the very path intended to save runtime.
        var nativeROITemporalSupport: [Float] = []
        if rendererProfile.isNativeROI {
            let x0 = rendererProfile.x
            let y0 = rendererProfile.y
            let x1 = x0 + rendererProfile.width
            let y1 = y0 + rendererProfile.height
            let feather = Float(max(rendererProfile.edgeFeather, 1))
            nativeROITemporalSupport = [Float](
                repeating: 0,
                count: rendererProfile.width * rendererProfile.height
            )
            let roiWidth = rendererProfile.width
            let roiHeight = rendererProfile.height
            let edgeFeather = rendererProfile.edgeFeather
            support.withUnsafeMutableBufferPointer { sup in
                nativeROITemporalSupport.withUnsafeMutableBufferPointer { roi in
                    let s = sup.baseAddress!
                    let t = roi.baseAddress!
                    Serve320ImageOps.fanOutLines(0, res - 1) { cy0, cy1 in
                        for yy in cy0...cy1 {
                            let row = yy * res
                            guard yy >= y0, yy < y1 else {
                                for xx in 0..<res { s[row + xx] = 0 }
                                continue
                            }
                            let localY = yy - y0
                            let yDistance = min(localY, roiHeight - 1 - localY)
                            let yRamp = edgeFeather == 0
                                ? Float(1) : min(Float(yDistance) / feather, 1)
                            for xx in 0..<res {
                                guard xx >= x0, xx < x1 else {
                                    s[row + xx] = 0
                                    continue
                                }
                                let localX = xx - x0
                                let localPixel = localY * roiWidth + localX
                                t[localPixel] = s[row + xx]
                                let xDistance = min(localX, roiWidth - 1 - localX)
                                let xRamp = edgeFeather == 0
                                    ? Float(1) : min(Float(xDistance) / feather, 1)
                                s[row + xx] *= xRamp * yRamp
                            }
                        }
                    }
                }
            }
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            stageTimings?.jawMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        var hostMouthUncoveredPixels = 0
        var jawProtectedSupportMax: Float = 0
        var jawProtectedMask = [UInt8](repeating: 0, count: plane)
        let statsLock = NSLock()
        support.withUnsafeBufferPointer { sup in
            hostMouth.withUnsafeBufferPointer { hostMouthBuf in
                hostFace.withUnsafeBufferPointer { face in
                    jawInterior.withUnsafeBufferPointer { jaw in
                        jawProtectedMask.withUnsafeMutableBufferPointer { mask in
                            let s = sup.baseAddress!
                            let hm = hostMouthBuf.baseAddress!
                            let fp = face.baseAddress!
                            let jp = jaw.baseAddress!
                            let mp = mask.baseAddress!
                            Serve320ImageOps.fanOutLines(0, res - 1) { cy0, cy1 in
                                var uncovered = 0
                                var localMax: Float = 0
                                for px in (cy0 * res)..<((cy1 + 1) * res) {
                                    if hm[px] >= 0.5, s[px] < 0.5 { uncovered += 1 }
                                    if fp[px] >= 0.5, jp[px] < 0.5 {
                                        localMax = max(localMax, s[px])
                                        if Float(px / res) >= hostCenter.y {
                                            mp[px] = 1
                                        }
                                    }
                                }
                                statsLock.lock()
                                hostMouthUncoveredPixels += uncovered
                                jawProtectedSupportMax = max(jawProtectedSupportMax, localMax)
                                statsLock.unlock()
                            }
                        }
                    }
                }
            }
        }
        let nativeROIHostFillApplied = rendererProfile.isNativeROI
        let temporalBoost = applyNativeROIHostFill(
            predCrop: &predCrop,
            support: &support,
            temporalActiveSupportROI: nativeROITemporalSupport,
            hostWarped: hostWarped,
            landmarks: tgtLm,
            frameID: hostFrameOffset + i)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            if var timings = stageTimings {
                timings.supportFinishMs = Self.milliseconds(from: stageMark, to: now)
                timings.supportMs = timings.apertureSupportMs + timings.capMs + timings.skinMs
                    + timings.targetMouthMs + timings.hostMouthMs + timings.supportCombineMs
                    + timings.jawMs + timings.supportFinishMs
                lastFrameStageTimings = timings
            }
        }

        return Serve320FrameOutput(index: i,
                                   rendererInferenceMs: rendererInferenceMs,
                                   predCropBGR: predCrop, support: support,
                                   box: bundle.stabBox(h), landmarks: tgtLm,
                                   aperture: ap, gate: gate, refRow: refRow,
                                   targetHostCenterDx: mouthCx - hostCenter.x,
                                   hostMouthRecoveredPixels: recoveredPixels,
                                   hostMouthUncoveredPixels: hostMouthUncoveredPixels,
                                   mouthInkShiftX: mouthInkShiftX,
                                   mouthInkRequestedShiftX: requestedMouthInkShift,
                                   mouthInkBoundsShiftX: mouthInkBoundsShift,
                                   mouthInkCentroidShiftX: mouthInkCentroidShift,
                                   mouthInkResidualDx: mouthInkResidualDx,
                                   rawMouthWidthToLandmarks: rawMouthWidthToLandmarks,
                                   rawMouthHeightToLandmarks: rawMouthHeightToLandmarks,
                                   rawMouthSharpness: rawMouthSharpness,
                                   rawMouthComponentCount: predInkMetrics?.componentCount,
                                   nativeROIHostFillApplied: nativeROIHostFillApplied,
                                   temporalBoostHandled: temporalBoost.handled,
                                   temporalBoostApplied: temporalBoost.effectApplied,
                                   jawProtectedSupportMax: jawProtectedSupportMax,
                                   jawProtectedMask: jawProtectedMask)
    }

    /// Eval-v2 scores the native ROI after host fill, while the historical app
    /// compositor expects a raw prediction plus a fractional support. Convert
    /// the ROI to that exact host-filled representation and return a binary
    /// ownership mask so the canvas compositor pastes it once. The optional
    /// Metal boost operates on this same representation, matching the frozen
    /// beta=.30 evaluator rather than accidentally multiplying support twice.
    @discardableResult
    private func applyNativeROIHostFill(
        predCrop: inout [UInt8],
        support: inout [Float],
        temporalActiveSupportROI: [Float],
        hostWarped: [UInt8],
        landmarks: [SIMD2<Float>],
        frameID: Int
    ) -> Serve320TemporalBoostApplication {
        guard rendererProfile.isNativeROI else { return .unavailable }
        let width = rendererProfile.width
        let height = rendererProfile.height
        let pixels = width * height
        guard temporalActiveSupportROI.count == pixels else { return .unavailable }
        var candidateROI = [UInt8](repeating: 0, count: pixels * 3)
        var hostROI = [UInt8](repeating: 0, count: pixels * 3)

        for localY in 0..<height {
            let globalY = rendererProfile.y + localY
            for localX in 0..<width {
                let globalX = rendererProfile.x + localX
                let localPixel = localY * width + localX
                let globalPixel = globalY * res + globalX
                let weight = min(max(support[globalPixel], 0), 1)
                let localBase = localPixel * 3
                let globalBase = globalPixel * 3
                for channel in 0..<3 {
                    let host = hostWarped[globalBase + channel]
                    hostROI[localBase + channel] = host
                    let value = Float(host) * (1 - weight)
                        + Float(predCrop[globalBase + channel]) * weight
                    candidateROI[localBase + channel] = UInt8(
                        min(max(value + 0.5, 0), 255))
                }
            }
        }

        let temporalBoost: Serve320TemporalBoostApplication
        if temporalBoostRequested {
            temporalBoost = temporalBoostSession.apply(
                candidateBGR: &candidateROI,
                hostBGR: hostROI,
                support: temporalActiveSupportROI,
                landmarks: landmarks,
                frameID: frameID,
                lease: temporalBoostLease
            )
        } else {
            temporalBoost = .unavailable
        }

        for localY in 0..<height {
            let globalY = rendererProfile.y + localY
            for localX in 0..<width {
                let globalX = rendererProfile.x + localX
                let localPixel = localY * width + localX
                let globalPixel = globalY * res + globalX
                let localBase = localPixel * 3
                let globalBase = globalPixel * 3
                predCrop[globalBase] = candidateROI[localBase]
                predCrop[globalBase + 1] = candidateROI[localBase + 1]
                predCrop[globalBase + 2] = candidateROI[localBase + 2]
                support[globalPixel] = support[globalPixel] > 1.0e-4 ? 1 : 0
            }
        }
        return temporalBoost
    }

    /// QA13/QA15 silence stack (Serve320Silence.swift) — set by product paths;
    /// nil for the frozen parity/T1 reference mirrors.
    var silenceEMA: Serve320SilenceEMA?

    /// Metal compute path for the paste + silence canvas ops
    /// (default intent plus the on-device parity self-check passed —
    /// AVATAR_METAL_COMPOSITE=0 disables it). nil = CPU reference
    /// loops, which stay the parity truth. The frozen reference mirrors
    /// (Serve320Parity, EvalT1) call canonicalCompositeNative directly and
    /// are never affected. Set before the first composited frame and never
    /// mid-clip: the two paths keep separate silence pixel state.
    var metalCompositor: Serve320MetalCompositor?

    /// Full composited 1080x1920 canvas for frame `i` (BGRA bytes).
    /// canonical_composite_native onto the decoded idle frame (replay.py:577).
    func compositedCanvas(frame i: Int) throws -> (bgra: [UInt8], out: Serve320FrameOutput) {
        let result = try makeCompositedCanvas(frame: i, measureVisual: false,
                                              preferMetalPresentation: false)
        return (result.bgra, result.out)
    }

    /// Eval-only final-pixel registration measurement. It compares the visible
    /// mouth component before and after the exact app compositor/silence stack.
    func compositedCanvasMeasured(frame i: Int) throws ->
        (bgra: [UInt8], out: Serve320FrameOutput,
         visual: Serve320CompositeVisualMetrics) {
        let result = try makeCompositedCanvas(frame: i, measureVisual: true,
                                              preferMetalPresentation: false)
        guard let visual = result.visual else {
            throw Serve320Error.invalidOutput("missing visual measurement")
        }
        return (result.bgra, result.out, visual)
    }

    private func makeCompositedCanvas(frame i: Int, measureVisual: Bool,
                                      preferMetalPresentation: Bool) throws ->
        (bgra: [UInt8], out: Serve320FrameOutput,
         visual: Serve320CompositeVisualMetrics?,
         metalSurface: Serve320MetalCompositor.PresentationSurface?) {
        let profiling = AvatarBenchmark.enabled
        var stageMark = profiling ? DispatchTime.now().uptimeNanoseconds : 0
        if profiling { lastFrameStageTimings = Serve320FrameStageTimings() }
        let h = hostFrameIndex(for: i)
        let cacheRepeatedHost = hostMotionPolicy.cachesRepeatedCanvasFrame
        let canvasBox = Serve320CanvasFrameBox()
        let canvasWork = DispatchWorkItem {
            let start = profiling ? DispatchTime.now().uptimeNanoseconds : 0
            do {
                canvasBox.bytes = try self.canvas.frameBytes(
                    h, cacheForReuse: cacheRepeatedHost)
            } catch {
                canvasBox.error = error
            }
            if profiling {
                canvasBox.decodeMs = Self.milliseconds(
                    from: start, to: DispatchTime.now().uptimeNanoseconds)
            }
        }
        canvasQueue.async(execute: canvasWork)

        // Idle-video decode/copy and renderer inference are independent. Run
        // them together, then preserve the old error precedence and exact bytes
        // before touching the canvas in the compositor.
        let renderResult = Result { try renderPredCrop(frame: i, hostFrame: h) }
        canvasWork.wait()
        if let error = canvasBox.error { throw error }
        guard var canvasBytes = canvasBox.bytes else {
            throw Serve320Error.canvasDecode(h)
        }
        let out = try renderResult.get()
        if profiling {
            lastFrameStageTimings?.canvasMs = canvasBox.decodeMs
            stageMark = DispatchTime.now().uptimeNanoseconds
        }
        let boxWidth = Float(out.box.x1 - out.box.x0)
        let boxHeight = Float(out.box.y1 - out.box.y0)
        let hostCenter = bundle.idleMouthCenter320(h)
        let expectedX = Float(out.box.x0)
            + (hostCenter.x + 0.5) * boxWidth / Float(res) - 0.5
        let expectedY = Float(out.box.y0)
            + (hostCenter.y + 0.5) * boxHeight / Float(res) - 0.5
        let idleCanvasBytes = measureVisual ? canvasBytes : []
        if profiling { stageMark = DispatchTime.now().uptimeNanoseconds }
        let jawSnapshot = jawProtectedSnapshot(
            canvas: canvasBytes,
            box: out.box,
            cropMask: out.jawProtectedMask)
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            lastFrameStageTimings?.jawSnapshotMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        let idleMetrics = measureVisual
            ? Serve320Compositor.mouthInkMetrics(
                imageBGRA: canvasBytes,
                width: Serve320Bundle.canvasW,
                height: Serve320Bundle.canvasH,
                expectedX: expectedX,
                expectedY: expectedY)
            : nil
        let silenceApplication = silenceEMA.map {
            Serve320MetalCompositor.SilenceApplication(
                ema: $0, frame: i, aperture: out.aperture)
        }
        var metalHandled = false
        var metalSurface: Serve320MetalCompositor.PresentationSurface?
        if let metal = metalCompositor {
            if preferMetalPresentation && !measureVisual {
                metalSurface = try metal.compositeFrameForPresentation(
                    canvas: canvasBytes, canvasW: Serve320Bundle.canvasW,
                    canvasH: Serve320Bundle.canvasH,
                    predBGR: out.predCropBGR, support: out.support, box: out.box,
                    jawRestore: jawSnapshot,
                    silence: silenceApplication)
                metalHandled = metalSurface != nil
            }
            if !metalHandled {
                metalHandled = metal.compositeFrame(
                    canvas: &canvasBytes, canvasW: Serve320Bundle.canvasW,
                    canvasH: Serve320Bundle.canvasH,
                    predBGR: out.predCropBGR, support: out.support, box: out.box,
                    silence: silenceApplication)
            }
            if profiling, let timings = metal.lastFrameTimings {
                lastFrameStageTimings?.metalUploadMs = timings.uploadMs
                lastFrameStageTimings?.metalEncodeMs = timings.encodeMs
                lastFrameStageTimings?.metalWaitMs = timings.waitMs
                lastFrameStageTimings?.metalGpuMs = timings.gpuMs
                lastFrameStageTimings?.metalReadbackMs = timings.readbackMs
            }
        }
        if !metalHandled {
            // CPU reference pair — the parity truth the Metal path is gated on.
            Serve320Compositor.canonicalCompositeNative(
                canvas: &canvasBytes, canvasW: Serve320Bundle.canvasW, canvasH: Serve320Bundle.canvasH,
                predBGR: out.predCropBGR, support: out.support, box: out.box)
            silenceEMA?.apply(frame: i, aperture: out.aperture,
                              canvas: &canvasBytes, canvasW: Serve320Bundle.canvasW)
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            lastFrameStageTimings?.compositeMs = Self.milliseconds(from: stageMark, to: now)
            stageMark = now
        }
        if metalSurface == nil {
            restoreJawProtectedSnapshot(jawSnapshot, canvas: &canvasBytes)
        }
        if profiling {
            let now = DispatchTime.now().uptimeNanoseconds
            lastFrameStageTimings?.jawRestoreMs = Self.milliseconds(from: stageMark, to: now)
        }
        let outputMetrics = measureVisual
            ? Serve320Compositor.mouthInkMetrics(
                imageBGRA: canvasBytes,
                width: Serve320Bundle.canvasW,
                height: Serve320Bundle.canvasH,
                expectedX: expectedX,
                expectedY: expectedY)
            : nil
        let jawProtection = measureVisual
            ? jawProtectedPixelDelta(
                before: idleCanvasBytes,
                after: canvasBytes,
                box: out.box,
                cropMask: out.jawProtectedMask)
            : (changed: 0, maxDelta: 0)
        let visual = measureVisual
            ? Serve320CompositeVisualMetrics(
                idleMouthCenterX: idleMetrics?.boundsCenterX,
                outputMouthCenterX: outputMetrics?.boundsCenterX,
                idleMouthCentroidX: idleMetrics?.centroidX,
                outputMouthCentroidX: outputMetrics?.centroidX,
                outputMouthComponentCount: outputMetrics?.componentCount,
                jawProtectedChangedPixels: jawProtection.changed,
                jawProtectedMaxChannelDelta: jawProtection.maxDelta)
            : nil
        return (canvasBytes, out, visual, metalSurface)
    }

    private func hostFrameIndex(for localSpeechFrame: Int) -> Int {
        hostMotionPolicy.frameIndex(
            forLocalSpeechFrame: localSpeechFrame,
            replyFrameOffset: hostFrameOffset,
            idleFrameCount: bundle.nIdle
        )
    }

    private func jawProtectedPixelDelta(
        before: [UInt8],
        after: [UInt8],
        box: (x0: Int, y0: Int, x1: Int, y1: Int),
        cropMask: [UInt8]
    ) -> (changed: Int, maxDelta: Int) {
        guard before.count == after.count else { return (0, 0) }
        let bw = box.x1 - box.x0
        let bh = box.y1 - box.y0
        guard bw > 0, bh > 0 else { return (0, 0) }
        var changed = 0
        var maxDelta = 0
        for y in 0..<bh {
            let cropY = min(max(Int((Float(y) + 0.5) * Float(res) / Float(bh)), 0), res - 1)
            for x in 0..<bw {
                let cropX = min(max(Int((Float(x) + 0.5) * Float(res) / Float(bw)), 0), res - 1)
                let cropPixel = cropY * res + cropX
                guard cropMask[cropPixel] != 0 else { continue }
                let canvasPixel = ((box.y0 + y) * Serve320Bundle.canvasW + box.x0 + x) * 4
                var pixelChanged = false
                for channel in 0..<3 {
                    let delta = abs(Int(after[canvasPixel + channel])
                                    - Int(before[canvasPixel + channel]))
                    maxDelta = max(maxDelta, delta)
                    pixelChanged = pixelChanged || delta != 0
                }
                if pixelChanged { changed += 1 }
            }
        }
        return (changed, maxDelta)
    }

    private func jawProtectedSnapshot(
        canvas: [UInt8],
        box: (x0: Int, y0: Int, x1: Int, y1: Int),
        cropMask: [UInt8]
    ) -> [Serve320MetalCompositor.JawRestorePixel] {
        let bw = box.x1 - box.x0
        let bh = box.y1 - box.y0
        guard bw > 0, bh > 0 else { return [] }
        var snapshot: [Serve320MetalCompositor.JawRestorePixel] = []
        for y in 0..<bh {
            let cropY = min(max(Int((Float(y) + 0.5) * Float(res) / Float(bh)), 0), res - 1)
            for x in 0..<bw {
                let cropX = min(max(Int((Float(x) + 0.5) * Float(res) / Float(bw)), 0), res - 1)
                guard cropMask[cropY * res + cropX] != 0 else { continue }
                let offset = ((box.y0 + y) * Serve320Bundle.canvasW + box.x0 + x) * 4
                let packedBGR = UInt32(canvas[offset])
                    | (UInt32(canvas[offset + 1]) << 8)
                    | (UInt32(canvas[offset + 2]) << 16)
                snapshot.append(.init(byteOffset: UInt32(offset),
                                      packedBGR: packedBGR))
            }
        }
        return snapshot
    }

    private func restoreJawProtectedSnapshot(
        _ snapshot: [Serve320MetalCompositor.JawRestorePixel],
        canvas: inout [UInt8]
    ) {
        for pixel in snapshot {
            let offset = Int(pixel.byteOffset)
            canvas[offset] = UInt8(pixel.packedBGR & 0xff)
            canvas[offset + 1] = UInt8((pixel.packedBGR >> 8) & 0xff)
            canvas[offset + 2] = UInt8((pixel.packedBGR >> 16) & 0xff)
        }
    }

    func compositedCGImage(frame i: Int) throws -> (image: CGImage, out: Serve320FrameOutput) {
        let frameStart = AvatarBenchmark.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        let (bytes, out) = try compositedCanvas(frame: i)
        let imageStart = AvatarBenchmark.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        guard let img = Self.bgraImage(bytes: bytes,
                                       w: Serve320Bundle.canvasW,
                                       h: Serve320Bundle.canvasH) else {
            throw Serve320Error.canvasDecode(i)
        }
        if AvatarBenchmark.enabled {
            let now = DispatchTime.now().uptimeNanoseconds
            lastFrameStageTimings?.imageMs = Self.milliseconds(from: imageStart, to: now)
            lastFrameStageTimings?.totalMs = Self.milliseconds(from: frameStart, to: now)
        }
        return (img, out)
    }

    /// Product frame entry point. Unless `AVATAR_METAL_PRESENT=0`, a verified
    /// Metal compositor returns a leased GPU texture directly to MetalFrameView.
    /// If that optional path cannot acquire a ring slot before any stateful
    /// work begins, this falls back to the established byte/CGImage path.
    func compositedFrame(frame i: Int) throws ->
        (frame: CompositedFrame, out: Serve320FrameOutput) {
        let frameStart = AvatarBenchmark.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        let result = try makeCompositedCanvas(
            frame: i, measureVisual: false,
            preferMetalPresentation: true)
        if let surface = result.metalSurface {
            if AvatarBenchmark.enabled {
                let now = DispatchTime.now().uptimeNanoseconds
                lastFrameStageTimings?.imageMs = 0
                lastFrameStageTimings?.totalMs = Self.milliseconds(from: frameStart, to: now)
            }
            return (CompositedFrame(
                source: nil,
                overlay: nil,
                metalSurface: surface,
                overlayRect: .zero,
                pixelWidth: Serve320Bundle.canvasW,
                pixelHeight: Serve320Bundle.canvasH,
                speechFrameIndex: hostFrameOffset + i,
                presentationTimeline: nil), result.out)
        }
        let imageStart = AvatarBenchmark.enabled ? DispatchTime.now().uptimeNanoseconds : 0
        guard let image = Self.bgraImage(bytes: result.bgra,
                                         w: Serve320Bundle.canvasW,
                                         h: Serve320Bundle.canvasH) else {
            throw Serve320Error.canvasDecode(i)
        }
        if AvatarBenchmark.enabled {
            let now = DispatchTime.now().uptimeNanoseconds
            lastFrameStageTimings?.imageMs = Self.milliseconds(from: imageStart, to: now)
            lastFrameStageTimings?.totalMs = Self.milliseconds(from: frameStart, to: now)
        }
        return (CompositedFrame(
            source: image,
            overlay: nil,
            metalSurface: nil,
            overlayRect: .zero,
            pixelWidth: Serve320Bundle.canvasW,
            pixelHeight: Serve320Bundle.canvasH,
            speechFrameIndex: hostFrameOffset + i,
            presentationTimeline: nil), result.out)
    }

    private static func milliseconds(from start: UInt64, to end: UInt64) -> Double {
        Double(end - start) / 1_000_000.0
    }

    // MARK: - image helpers

    /// Planar BGR u8 (3,res,res) -> HWC BGR (res,res,3) — cv2 image layout.
    static func planarBGRToHWC(_ planar: UnsafeRawBufferPointer, res: Int) -> [UInt8] {
        let plane = res * res
        var out = [UInt8](repeating: 0, count: plane * 3)
        for px in 0..<plane {
            out[px * 3] = planar[px]                  // B
            out[px * 3 + 1] = planar[plane + px]      // G
            out[px * 3 + 2] = planar[2 * plane + px]  // R
        }
        return out
    }

    /// Planar BGR u8 -> planar RGB01 (render_utils.bgr_u8_to_rgb01:233-236: flip+float/255).
    static func planarBGRToRGB01(_ planar: UnsafeRawBufferPointer, res: Int) -> [Float] {
        let plane = res * res
        var out = [Float](repeating: 0, count: plane * 3)
        for px in 0..<plane {
            out[px] = Float(planar[2 * plane + px]) / 255          // R plane first
            out[plane + px] = Float(planar[plane + px]) / 255      // G
            out[2 * plane + px] = Float(planar[px]) / 255          // B
        }
        return out
    }

    // MARK: render_utils.chin_warp (arm_a/render_utils.py:287-325)
    /// QA6 R2: aperture-scaled DOWNWARD warp of the host chin band. Deterministic
    /// vertical remap, INTER_LINEAR, BORDER_REPLICATE; byte-passthrough when
    /// aperture <= closed (d < 0.5px). Applied at 320 with max_warp 16, ramp 28,
    /// mouth_floor = int(0.62*320) = 198 (serve convention — NOT rescaled).
    static func chinWarp(_ cropHWC: [UInt8], contourU8: UnsafeRawBufferPointer,
                         aperture: Float, mouthFloor: Int,
                         maxWarpPx: Float = 16, rampPx: Float = 28,
                         closedAp: Float = 0.20, openAp: Float = 0.80) -> [UInt8] {
        let H = Serve320Bundle.res, W = Serve320Bundle.res
        // :306-309
        let frac = min(max((aperture - closedAp) / max(openAp - closedAp, 1e-6), 0), 1)
        let d = frac * maxWarpPx
        if d < 0.5 { return cropHWC }
        // :313-314 — chin-bottom row = last row with any contour px (fc >= 0.5 <=> u8 >= 128)
        var yC = H - 1
        outer: for y in stride(from: H - 1, through: 0, by: -1) {
            let row = y * W
            for x in 0..<W {
                if contourU8[row + x] >= 128 { yC = y; break outer }
            }
        }
        // :317-319 — ramp starts ramp_px ABOVE the chin, never above the mouth floor
        let yA = max(mouthFloor, yC - Int(rampPx))
        if yA >= yC { return cropHWC }
        var out = cropHWC
        // :320-325 — dy ramps 0 -> d from y_a to the chin, held d below (collar);
        // sample from above (map_y = clip(ys - dy, 0, H-1)); rows < y_a untouched.
        for y in yA..<H {
            let rampDenom = Float(max(yC - yA, 1))
            let ramp = min(max(Float(y - yA) / rampDenom, Float(0)), Float(1))
            let dy = d * ramp
            let srcY = min(max(Float(y) - dy, Float(0)), Float(H - 1))
            let y0 = srcY.rounded(FloatingPointRoundingRule.down)
            let fy = srcY - y0
            let iy0 = Int(y0), iy1 = min(iy0 + 1, H - 1)
            for x in 0..<W {
                let base0 = (iy0 * W + x) * 3
                let base1 = (iy1 * W + x) * 3
                let dst = (y * W + x) * 3
                for c in 0..<3 {
                    // cv2.remap INTER_LINEAR on u8: fixed-point round-half-up
                    let v = Float(cropHWC[base0 + c]) * (1 - fy) + Float(cropHWC[base1 + c]) * fy
                    out[dst + c] = UInt8(min((v + 0.5).rounded(.down), 255))
                }
            }
        }
        return out
    }

    /// BGRA bytes -> CGImage (VideoSource.cgImage bitmap conventions).
    static func bgraImage(bytes: [UInt8], w: Int, h: Int) -> CGImage? {
        let cs = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.union(
            CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedFirst.rawValue))
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: bitmapInfo.rawValue),
              let data = ctx.data else { return nil }
        _ = bytes.withUnsafeBytes { raw in
            memcpy(data, raw.baseAddress, w * h * 4)
        }
        return ctx.makeImage()
    }

    /// Pay the renderer's cold-start cost BEFORE a turn instead of during it.
    ///
    /// Measured on an iPhone Air, one realtime reply (avatar_bench.log):
    ///
    ///     serve320_first_frame     render_ms=77.1   <- chunk 0, first frame
    ///     serve320_first_frame     render_ms=59.7   <- chunk 1, first frame
    ///     serve320_render_complete avg_render_ms=33.0
    ///
    /// A 2.3x cliff on the first frame of every chunk, against a 40 ms frame
    /// budget. It lands exactly where it hurts: stream playout starts on the
    /// first audio delta, so the reply is already audible while the renderer is
    /// still paying it, and the loop's skip-ahead branch then drops the frames
    /// it is late for — 8 frames dropped across that turn. That is the "lips
    /// don't move for the first moment of every new turn" report.
    ///
    /// Called on `response.created`, which the same log puts 485 ms ahead of
    /// first playout, so the cliff is absorbed in dead time.
    ///
    /// Allocates its own input rather than reusing `rendererInputArray()`: that
    /// buffer is shared with the live render loop, and a warm that overlaps a
    /// running chunk must not scribble on the frame being composited. Zeros are
    /// fine — the cost being paid here is graph compilation and ANE residency,
    /// neither of which depends on the values.
    /// Static because the thing that goes cold is the MODEL, and the model
    /// outlives any one pipeline: `Serve320TextChain` holds it for the whole
    /// call while a fresh `Serve320Pipeline` is built per prepared chunk.
    static func warmRenderer(
        _ renderer: MLModel,
        profile: Serve320RendererProfile = .full320,
        iterations: Int = 2
    ) {
        guard iterations > 0,
              let input = try? MLMultiArray(
                  shape: [
                      1,
                      27,
                      NSNumber(value: profile.height),
                      NSNumber(value: profile.width),
                  ],
                  dataType: .float32),
              let provider = try? MLDictionaryFeatureProvider(dictionary: [
                  "x": MLFeatureValue(multiArray: input),
              ])
        else { return }
        for _ in 0..<iterations {
            _ = try? renderer.prediction(from: provider)
        }
    }

    func warmRenderer(iterations: Int = 2) {
        Self.warmRenderer(renderer, profile: rendererProfile, iterations: iterations)
    }

    /// Copy the exact NCHW tensor most recently submitted to the renderer.
    ///
    /// This is intentionally an internal QA hook rather than a product output.
    /// Serve320Parity uses it only when the explicit calibration-dump switch is
    /// enabled, so normal playback never pays the 10.5 MiB copy per frame.
    func rendererInputSnapshot() throws -> [Float] {
        guard let rendererInput else {
            throw Serve320Error.invalidOutput("renderer input before first frame")
        }
        guard rendererInput.dataType == .float32 else {
            throw Serve320Error.invalidOutput(
                "renderer input dtype \(rendererInput.dataType.rawValue)")
        }
        let pointer = rendererInput.dataPointer.bindMemory(
            to: Float.self,
            capacity: rendererInput.count)
        return Array(UnsafeBufferPointer(start: pointer, count: rendererInput.count))
    }

    private func rendererInputArray() throws -> MLMultiArray {
        if let rendererInput { return rendererInput }
        let arr = try MLMultiArray(
            shape: [
                1,
                27,
                NSNumber(value: rendererProfile.height),
                NSNumber(value: rendererProfile.width),
            ],
            dataType: .float32)
        rendererInput = arr
        return arr
    }
}

// MARK: - Idle canvas reader (sequential AVAssetReader; VideoSource.swift pattern,
// but returns raw BGRA bytes so the QA9 compositor can write into the box region)

final class Serve320CanvasReader {
    private var asset: AVURLAsset?
    private var track: AVAssetTrack?
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var readHead = 0
    private var cachedFrameIndex: Int?
    private var cachedFrameBytes: [UInt8]?
    /// Realtime prepares several geometry windows ahead. Their render loops are
    /// normally sequential, but cancellation can briefly overlap old and new
    /// generations. A shared reader removes the O(reply offset) decode restart;
    /// this lock keeps that shared AVAssetReader correct during handoff.
    private let readLock = NSLock()
    let width = Serve320Bundle.canvasW
    let height = Serve320Bundle.canvasH

    private func ensureTrack() async throws {
        if track != nil { return }
        guard let url = Serve320Bundle.defaultRoot()?
            .appendingPathComponent("idle320_loop.mp4") else {
            throw Serve320Error.bundleMissing
        }
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let t = tracks.first else {
            throw Serve320Error.canvasDecode(-1)
        }
        self.asset = asset
        self.track = t
    }

    /// Async one-time init (AVAssetTrack loading is async on iOS 17+).
    func prepare() async throws {
        try await ensureTrack()
    }

    private func openReader() throws {
        reader?.cancelReading()
        guard let asset, let track else {
            throw Serve320Error.canvasDecode(-2)
        }
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        ]
        let newReader = try AVAssetReader(asset: asset)
        let newOutput = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        newOutput.alwaysCopiesSampleData = false
        newReader.add(newOutput)
        guard newReader.startReading() else {
            throw Serve320Error.canvasDecode(-3)
        }
        reader = newReader
        output = newOutput
        readHead = 0
    }

    /// Frame `idx` as BGRA bytes (width*height*4). Sequential access reopens
    /// the reader when the index rewinds. Fixed-host product playback retains
    /// one decoded frame so it does not decode the same HEVC sample 25 times/s.
    /// Call `prepare()` once before first use.
    func frameBytes(_ idx: Int, cacheForReuse: Bool = false) throws -> [UInt8] {
        readLock.lock()
        defer { readLock.unlock() }
        guard track != nil else {
            throw Serve320Error.canvasDecode(-4)
        }
        if cacheForReuse,
           cachedFrameIndex == idx,
           let cachedFrameBytes {
            return cachedFrameBytes
        }
        if !cacheForReuse {
            cachedFrameIndex = nil
            cachedFrameBytes = nil
        }
        if reader == nil || idx < readHead {
            try openReader()
        }
        while readHead <= idx {
            guard let sample = output?.copyNextSampleBuffer() else {
                try openReader()   // EOF safety; reopen once
                continue
            }
            let thisIdx = readHead
            readHead += 1
            guard thisIdx == idx else { continue }
            guard let buf = CMSampleBufferGetImageBuffer(sample) else {
                throw Serve320Error.canvasDecode(idx)
            }
            CVPixelBufferLockBaseAddress(buf, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
            let w = CVPixelBufferGetWidth(buf)
            let h = CVPixelBufferGetHeight(buf)
            let bpr = CVPixelBufferGetBytesPerRow(buf)
            guard let base = CVPixelBufferGetBaseAddress(buf) else {
                throw Serve320Error.canvasDecode(idx)
            }
            var out = [UInt8](repeating: 255, count: width * height * 4)
            let copyW = min(w, width) * 4
            out.withUnsafeMutableBytes { dstRaw in
                for y in 0..<min(h, height) {
                    memcpy(dstRaw.baseAddress! + y * width * 4,
                           base + y * bpr, copyW)
                }
            }
            if cacheForReuse {
                cachedFrameIndex = idx
                cachedFrameBytes = out
            }
            return out
        }
        throw Serve320Error.canvasDecode(idx)
    }
}
