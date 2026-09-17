//
//  Serve320MetalCompositor.swift
//  Metal compute path for the Serve320 per-frame canvas ops (paste + QA13
//  silence EMA + QA15 hard-lock), kernels in Compositor.metal. Gated twice:
//    1. AVATAR_METAL_COMPOSITE is not "0" (default ON; "0" is the rollback),
//       and
//    2. a one-time on-device CPU-vs-Metal self-check (Serve320MetalParity,
//       max abs diff <= 1 over random frames) whose pass is persisted in
//       UserDefaults; until it has passed, the flagged mode still runs CPU.
//
//  The library is compiled AT RUNTIME from the bundled Compositor.metal
//  source with fast math OFF (MTLMathMode.safe). Xcode's default .metal
//  pipeline compiles with fast math ON, which permits FMA contraction —
//  1-ulp perturbations that can flip the half-even roundings the CPU
//  reference pins down. Runtime compilation gives the app and the
//  standalone harness (scripts/serve320_metal_parity.swift) one identical
//  compilation path, and the self-check gates exactly the library that will
//  serve frames. Compile cost is paid once per process (cached MTLLibrary).
//
//  Buffers are allocated once and reused across frames (canvas/pred/support
//  at init, ROI-sized silence state on EMA-owner change — a per-utterance
//  event). Only the rows the kernels touch (stab box + silence ROI) are
//  copied host<->GPU each frame. Instances are not internally locked: canvas
//  compositors stay pipeline-local and render serially, while the dedicated
//  shared temporal runtime is serialized by Serve320TemporalBoostSession.
//

import Foundation
import Metal
import QuartzCore

final class Serve320MetalCompositor {

    typealias Telemetry = @Sendable (_ event: String, _ detail: String) -> Void

    struct SilenceApplication {
        let ema: Serve320SilenceEMA
        let frame: Int
        let aperture: Float
    }

    struct FrameTimings: Sendable {
        let uploadMs: Double
        let encodeMs: Double
        let waitMs: Double
        let gpuMs: Double
        let readbackMs: Double
    }

    /// Compact byte-exact jaw restore record. `packedBGR` stores B, G, and R
    /// in bits 0...23; alpha is deliberately left untouched, matching the CPU
    /// reference restore in Serve320Pipeline.
    struct JawRestorePixel {
        let byteOffset: UInt32
        let packedBGR: UInt32
    }

    /// A leased output texture from the three-slot presentation ring. The
    /// compositor never reuses the slot until every owner (including the
    /// Metal command-buffer completion handler) releases this object.
    final class PresentationSurface: @unchecked Sendable {
        let texture: MTLTexture
        let readyEvent: MTLSharedEvent
        let readyValue: UInt64
        private let producerCommandBuffer: MTLCommandBuffer
        private let owner: Serve320MetalCompositor
        private let slot: Int

        fileprivate init(texture: MTLTexture,
                         readyEvent: MTLSharedEvent,
                         readyValue: UInt64,
                         producerCommandBuffer: MTLCommandBuffer,
                         owner: Serve320MetalCompositor,
                         slot: Int) {
            self.texture = texture
            self.readyEvent = readyEvent
            self.readyValue = readyValue
            self.producerCommandBuffer = producerCommandBuffer
            self.owner = owner
            self.slot = slot
        }

        /// Valid after a consumer command buffer has passed `readyValue`.
        var producerGpuMs: Double {
            producerCommandBuffer.gpuEndTime > producerCommandBuffer.gpuStartTime
                ? (producerCommandBuffer.gpuEndTime
                    - producerCommandBuffer.gpuStartTime) * 1_000.0
                : 0
        }

        deinit { owner.releasePresentationSlot(slot) }
    }

    /// Per-slot producer inputs. Shared CPU buffers cannot be reused until the
    /// GPU has consumed them; tying them to the leased output slot makes the
    /// no-wait path race-free even when several frames are in flight.
    private final class PresentationSlot {
        let texture: MTLTexture
        let canvas: MTLBuffer
        let prediction: MTLBuffer
        let support: MTLBuffer
        var jawRestore: MTLBuffer?

        init(texture: MTLTexture, canvas: MTLBuffer,
             prediction: MTLBuffer, support: MTLBuffer) {
            self.texture = texture
            self.canvas = canvas
            self.prediction = prediction
            self.support = support
        }
    }

    static let flagEnvVar = "AVATAR_METAL_COMPOSITE"
    /// Default-on end-to-end presentation path. A same-binary physical-device
    /// A/B measured 13.7 -> 11.8 ms mean and 15.7 -> 13.3 ms p95 across 500
    /// frames per arm, with zero skips or missed 40 ms budgets. `=0` preserves
    /// the established readback/CGImage path as an immediate rollback.
    static let presentationFlagEnvVar = "AVATAR_METAL_PRESENT"
    static var presentationFlagEnabled: Bool {
        ProcessInfo.processInfo.environment[presentationFlagEnvVar] != "0"
    }
    /// Default ON since 2026-07-26. The flag became opt-*out* once the kernels
    /// measured bit-identical to the CPU truth rather than merely within the
    /// <=1 tolerance: 86 frames across all five parity scenarios
    /// (paste-identity-halfeven, paste-resample, silence-hardlock,
    /// silence-ema-moving, full-canvas-hardlock) at **max abs diff 0**, and
    /// 2.23 -> 0.60 ms/frame including host<->GPU transfers.
    ///
    /// This only expresses intent. The on-device parity self-check in
    /// `createVerifiedForProduct` is still the gate that decides, and it falls
    /// back to the CPU loops on any device where the check has not passed —
    /// so a GPU whose rounding differs from this one silently keeps the
    /// canonical path rather than shipping wrong pixels.
    static var flagEnabled: Bool {
        ProcessInfo.processInfo.environment[flagEnvVar] != "0"
    }
    /// Bump the suffix whenever kernel semantics change so stale passes
    /// cannot vouch for new code.
    static let parityDefaultsKey = "Serve320MetalCompositeParity_v1"

    // Must mirror the MSL structs in Compositor.metal (int/float, no padding).
    private struct PasteParams {
        var canvasW: Int32, boxX0: Int32, boxY0: Int32, boxW: Int32, boxH: Int32
        var srcRes: Int32
        var scaleX: Float, scaleY: Float
    }
    private struct RoiParams {
        var canvasW: Int32, roiX0: Int32, roiY0: Int32, roiW: Int32, roiH: Int32
        var beta: Float
    }
    private struct TemporalBoostParams {
        var roi: SIMD4<Int32>
        var warpBounds: SIMD4<Int32>
        var flags: Int32
        var beta: Float
        var deltaClip: Float
        var warpSupport: Float
        var coreBounds: SIMD4<Float>
    }

    /// Turn/stream-scoped state for the isolated native-ROI articulation
    /// primitive. The textures contain only unboosted candidate-minus-host
    /// residuals; boosted pixels can never feed back into the next frame.
    final class TemporalBoostState {
        fileprivate var previousResidual: MTLTexture
        fileprivate var nextResidual: MTLTexture
        fileprivate let previousLandmarks: MTLBuffer
        fileprivate var previousPointValues: [SIMD2<Float>]
        fileprivate var hasHistory = false
        fileprivate var lastFrameID: Int?

        fileprivate init(previousResidual: MTLTexture, nextResidual: MTLTexture,
                         previousLandmarks: MTLBuffer) {
            self.previousResidual = previousResidual
            self.nextResidual = nextResidual
            self.previousLandmarks = previousLandmarks
            self.previousPointValues = [SIMD2<Float>](repeating: .zero, count: 20)
        }

        /// Call for a new response, confirmed interruption/tombstone,
        /// teardown, identity/renderer change, or seek. A nonconsecutive frame
        /// ID also resets automatically in `applyTemporalBoostFrame`.
        func reset() {
            hasHistory = false
            lastFrameID = nil
        }
    }

    private final class TemporalBoostResources {
        let pso: MTLComputePipelineState
        let candidate: MTLBuffer
        let host: MTLBuffer
        let support: MTLBuffer
        let currentLandmarks: MTLBuffer
        let changed: MTLBuffer

        init(pso: MTLComputePipelineState, candidate: MTLBuffer, host: MTLBuffer,
             support: MTLBuffer, currentLandmarks: MTLBuffer,
             changed: MTLBuffer) {
            self.pso = pso
            self.candidate = candidate
            self.host = host
            self.support = support
            self.currentLandmarks = currentLandmarks
            self.changed = changed
        }
    }

    private struct CanvasParams {
        var canvasW: Int32, canvasH: Int32
    }
    private struct JawRestoreParams {
        var count: UInt32
    }

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let library: MTLLibrary
    private let pastePSO: MTLComputePipelineState
    private let roiCopyPSO: MTLComputePipelineState
    private let emaPSO: MTLComputePipelineState
    private let lockBlendPSO: MTLComputePipelineState
    private let emaResetPSO: MTLComputePipelineState
    private let jawRestorePSO: MTLComputePipelineState?
    private let canvasToTexturePSO: MTLComputePipelineState?

    private let res = 320
    private let predBuf: MTLBuffer      // 320*320*3 u8, reused every frame
    private let supportBuf: MTLBuffer   // 320*320 f32, reused every frame
    private var canvasBuf: MTLBuffer?   // canvasW*canvasH*4 u8, sized on first frame
    private(set) var lastFrameTimings: FrameTimings?
    private let presentationLock = NSLock()
    private static let presentationCanvasW = 1080
    private static let presentationCanvasH = 1920
    private let presentationEvent: MTLSharedEvent?
    private var presentationSlots: [PresentationSlot] = []
    private var presentationLeased = [Bool](repeating: false, count: 3)
    private var nextPresentationSlot = 0
    private var nextPresentationSignalValue: UInt64 = 1
    private var presentationTelemetry: Telemetry?

    // Silence ROI state, owned per Serve320SilenceEMA instance (recreated on
    // owner change — a per-utterance event, never per-frame). The owner is
    // held STRONGLY: keying by ObjectIdentifier alone let a freshly allocated
    // EMA reuse a dead one's address and silently inherit the previous
    // utterance's ROI/mask/state buffers.
    private var silenceOwner: Serve320SilenceEMA?
    private var silenceRoi = (x0: 0, y0: 0, x1: 0, y1: 0)
    private var maskBuf: MTLBuffer?     // hh*ww f32 ramp, uploaded once per owner
    private var regBuf: MTLBuffer?      // hh*ww*4 u8 pre-blend capture
    private var emaBuf: MTLBuffer?      // hh*ww*4 f32 EMA state
    private var lockBuf: MTLBuffer?     // hh*ww*4 u8 hard-lock snapshot

    // Lazily allocated so the unintegrated experiment changes neither memory
    // footprint nor dispatches in the shipping compositor path.
    private var temporalBoostResources: TemporalBoostResources?

    static let temporalBoostROI = (x: 48, y: 96, width: 224, height: 160)
    // Frozen full32 keeper for the base32 native-ROI production renderer.
    // Raw base32 misses silence flicker; beta=.20 passes all 9/9 gates.
    static let temporalBoostBeta: Float = 0.20
    static let temporalBoostDeltaClip: Float = 12.0
    static let temporalBoostWarpSupport: Float = 24.0

    // MARK: - creation

    private nonisolated(unsafe) static var cachedLibrary: MTLLibrary?
    private nonisolated(unsafe) static var verifiedThisProcess = false
    private static let creationLock = NSLock()

    /// Compositor.metal ships as a bundle RESOURCE (raw source), not through
    /// Xcode's metallib pipeline — see the header for why.
    static func loadBundledShaderSource() -> String? {
        guard let url = YoobResources.url(forResource: "Compositor",
                                        withExtension: "metal") else { return nil }
        return try? String(contentsOf: url, encoding: .utf8)
    }

    /// Product entry point (Serve320Player / Serve320TextChain): nil unless
    /// the env flag is set, Metal + shader compile succeed, AND the parity
    /// self-check has passed on this device (now or on a prior launch).
    /// Eval/parity reference mirrors never call this — they stay CPU.
    /// `--avatar-demo-metal-bench` runs the full parity suite and the CPU-vs-Metal
    /// benchmark on the real device and prints both, then leaves the product path
    /// untouched. The host Mac's GPU is not the phone's, so shipping decisions
    /// about the compositor need numbers from the device itself.
    static let benchArgument = "--avatar-demo-metal-bench"
    static var benchRequested: Bool {
        ProcessInfo.processInfo.arguments.contains(benchArgument)
    }

    static func runDeviceBenchmark(_ metal: Serve320MetalCompositor) {
        let report = Serve320MetalParity.run(metal: metal, quick: false)
        print("[Serve320Metal] device parity: \(report.detail)")
        print("[Serve320Metal] device parity \(report.pass ? "PASS" : "FAIL") "
              + "(\(report.framesCompared) frames, max abs diff \(report.maxAbsDiff))")
        let bench = Serve320MetalParity.benchmark(metal: metal)
        let speedup = bench.metalMs > 0 ? bench.cpuMs / bench.metalMs : 0
        print(String(format: "[Serve320Metal] DEVICE BENCH cpu %.3f ms/frame · metal %.3f ms/frame "
                     + "· %.2fx (incl. transfers)", bench.cpuMs, bench.metalMs, speedup))
    }

    static func createVerifiedForProduct(
        telemetry: Telemetry? = nil
    ) -> Serve320MetalCompositor? {
        guard flagEnabled else { return nil }
        guard let source = loadBundledShaderSource() else {
            print("[Serve320Metal] Compositor.metal missing from bundle — CPU path")
            return nil
        }
        guard let metal = Serve320MetalCompositor(shaderSource: source) else {
            print("[Serve320Metal] device/compile/pipeline init failed — CPU path")
            return nil
        }
        metal.presentationTelemetry = telemetry
        creationLock.lock()
        let alreadyVerified = verifiedThisProcess
            || UserDefaults.standard.bool(forKey: parityDefaultsKey)
        creationLock.unlock()
        // A persisted pass short-circuits the check on every later launch, which
        // would also skip the benchmark — so an explicit bench request overrides it.
        if alreadyVerified && !benchRequested { return metal }
        if benchRequested {
            runDeviceBenchmark(metal)
            return metal
        }
        let report = Serve320MetalParity.run(metal: metal, quick: true)
        guard report.pass else {
            print("[Serve320Metal] parity self-check FAIL — CPU path. \(report.detail)")
            return nil
        }
        creationLock.lock()
        verifiedThisProcess = true
        UserDefaults.standard.set(true, forKey: parityDefaultsKey)
        creationLock.unlock()
        print("[Serve320Metal] parity self-check PASS (max abs diff \(report.maxAbsDiff)) — Metal path enabled")
        if benchRequested { runDeviceBenchmark(metal) }
        return metal
    }

    init?(shaderSource: String) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { return nil }
        Self.creationLock.lock()
        var library = Self.cachedLibrary
        if library == nil || library?.device !== device {
            let options = MTLCompileOptions()
            if #available(iOS 18.0, macOS 15.0, *) {
                options.mathMode = .safe
            } else {
                options.fastMathEnabled = false
            }
            library = try? device.makeLibrary(source: shaderSource, options: options)
            Self.cachedLibrary = library
        }
        Self.creationLock.unlock()
        guard let library else { return nil }
        func pso(_ name: String) -> MTLComputePipelineState? {
            guard let fn = library.makeFunction(name: name) else { return nil }
            return try? device.makeComputePipelineState(function: fn)
        }
        let presentationEnabled = Self.presentationFlagEnabled
        let jawRestore = presentationEnabled ? pso("serve320_jaw_restore") : nil
        let canvasToTexture = presentationEnabled ? pso("serve320_canvas_to_texture") : nil
        guard let paste = pso("serve320_paste"),
              let roiCopy = pso("serve320_roi_copy"),
              let emaApply = pso("serve320_silence_ema"),
              let lockBlend = pso("serve320_silence_lock_blend"),
              let emaReset = pso("serve320_ema_reset"),
              let pred = device.makeBuffer(length: res * res * 3,
                                           options: .storageModeShared),
              let support = device.makeBuffer(length: res * res * MemoryLayout<Float>.stride,
                                              options: .storageModeShared) else {
            return nil
        }
        if presentationEnabled && (jawRestore == nil || canvasToTexture == nil) {
            return nil
        }
        self.device = device
        self.queue = queue
        self.library = library
        self.pastePSO = paste
        self.roiCopyPSO = roiCopy
        self.emaPSO = emaApply
        self.lockBlendPSO = lockBlend
        self.emaResetPSO = emaReset
        self.jawRestorePSO = jawRestore
        self.canvasToTexturePSO = canvasToTexture
        self.predBuf = pred
        self.supportBuf = support
        self.presentationEvent = presentationEnabled ? device.makeSharedEvent() : nil

        if presentationEnabled {
            guard presentationEvent != nil else { return nil }
            let desc = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm,
                width: Self.presentationCanvasW,
                height: Self.presentationCanvasH,
                mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            desc.storageMode = .private
            for _ in 0..<presentationLeased.count {
                guard let texture = device.makeTexture(descriptor: desc),
                      let canvas = device.makeBuffer(
                        length: Self.presentationCanvasW
                            * Self.presentationCanvasH * 4,
                        options: .storageModeShared),
                      let prediction = device.makeBuffer(
                        length: res * res * 3,
                        options: .storageModeShared),
                      let support = device.makeBuffer(
                        length: res * res * MemoryLayout<Float>.stride,
                        options: .storageModeShared) else {
                    return nil
                }
                presentationSlots.append(PresentationSlot(
                    texture: texture,
                    canvas: canvas,
                    prediction: prediction,
                    support: support))
            }
        }
    }

    // MARK: - isolated native-ROI temporal articulation experiment

    /// Allocate independent state for one renderer stream/turn. Product code
    /// reaches this only through Serve320TemporalBoostSession, which owns the
    /// lifecycle lease and serializes access across streamed chunk pipelines.
    func makeTemporalBoostState() -> TemporalBoostState? {
        guard ensureTemporalBoostResources() != nil else { return nil }
        let roi = Self.temporalBoostROI
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba32Float, width: roi.width, height: roi.height,
            mipmapped: false)
        descriptor.storageMode = .private
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let previous = device.makeTexture(descriptor: descriptor),
              let next = device.makeTexture(descriptor: descriptor),
              let landmarks = device.makeBuffer(
                length: 20 * MemoryLayout<SIMD2<Float>>.stride,
                options: .storageModeShared) else { return nil }
        return TemporalBoostState(previousResidual: previous,
                                  nextResidual: next,
                                  previousLandmarks: landmarks)
    }

    /// Apply the selected beta=.20 residual boost to one packed 224x160 BGR
    /// ROI. Landmarks are the 20 mouth points in full-320 coordinates. The
    /// first/reset/discontinuous frame and beta==0 return `candidateBGR`
    /// byte-for-byte while still replacing state with the current raw residual.
    @discardableResult
    func applyTemporalBoostFrame(candidateBGR: inout [UInt8], hostBGR: [UInt8],
                                 support: [Float], landmarks: [SIMD2<Float>],
                                 frameID: Int, reset: Bool,
                                 beta: Float = Serve320MetalCompositor.temporalBoostBeta,
                                 state: TemporalBoostState)
        -> Serve320TemporalBoostApplication {
        let roi = Self.temporalBoostROI
        let pixels = roi.width * roi.height
        guard candidateBGR.count == pixels * 3,
              hostBGR.count == pixels * 3,
              support.count == pixels,
              landmarks.count == 20,
              beta.isFinite, beta >= 0, beta <= 1,
              landmarks.allSatisfy({ $0.x.isFinite && $0.y.isFinite }),
              state.previousResidual.device === device,
              let resources = ensureTemporalBoostResources(),
              MemoryLayout<TemporalBoostParams>.stride == 64 else {
            return .unavailable
        }

        let consecutive: Bool
        if let last = state.lastFrameID, last < Int.max {
            consecutive = frameID == last + 1
        } else {
            consecutive = false
        }
        let useHistory = !reset && state.hasHistory && consecutive
        let sourcePoints = useHistory ? state.previousPointValues : landmarks
        var hasMotion = false
        if useHistory {
            for index in 0..<20 {
                let d = landmarks[index] - sourcePoints[index]
                if d.x * d.x + d.y * d.y > 1.0e-12 {
                    hasMotion = true
                    break
                }
            }
        }

        var minX = sourcePoints[0].x, minY = sourcePoints[0].y
        var maxX = minX, maxY = minY
        for point in sourcePoints + landmarks {
            minX = min(minX, point.x)
            minY = min(minY, point.y)
            maxX = max(maxX, point.x)
            maxY = max(maxY, point.y)
        }
        let warp = Self.temporalBoostWarpSupport
        let x0 = max(0, Int(floor(Double(minX - warp))))
        let y0 = max(0, Int(floor(Double(minY - warp))))
        let x1 = min(res, Int(ceil(Double(maxX + warp + 1))))
        let y1 = min(res, Int(ceil(Double(maxY + warp + 1))))
        var flags: Int32 = useHistory ? 1 : 0
        if hasMotion { flags |= 2 }
        var params = TemporalBoostParams(
            roi: SIMD4(Int32(roi.x), Int32(roi.y), Int32(roi.width), Int32(roi.height)),
            warpBounds: SIMD4(Int32(x0), Int32(y0), Int32(x1), Int32(y1)),
            flags: flags,
            beta: beta,
            deltaClip: Self.temporalBoostDeltaClip,
            warpSupport: warp,
            coreBounds: SIMD4(minX, minY, maxX, maxY))

        candidateBGR.withUnsafeBytes {
            _ = memcpy(resources.candidate.contents(), $0.baseAddress!, $0.count)
        }
        hostBGR.withUnsafeBytes {
            _ = memcpy(resources.host.contents(), $0.baseAddress!, $0.count)
        }
        support.withUnsafeBytes {
            _ = memcpy(resources.support.contents(), $0.baseAddress!, $0.count)
        }
        landmarks.withUnsafeBytes {
            _ = memcpy(resources.currentLandmarks.contents(), $0.baseAddress!, $0.count)
        }
        resources.changed.contents().assumingMemoryBound(to: UInt32.self).pointee = 0

        guard let command = queue.makeCommandBuffer(),
              let encoder = command.makeComputeCommandEncoder() else {
            return .unavailable
        }
        encoder.setComputePipelineState(resources.pso)
        encoder.setBuffer(resources.candidate, offset: 0, index: 0)
        encoder.setBuffer(resources.host, offset: 0, index: 1)
        encoder.setBuffer(resources.support, offset: 0, index: 2)
        encoder.setBuffer(state.previousLandmarks, offset: 0, index: 3)
        encoder.setBuffer(resources.currentLandmarks, offset: 0, index: 4)
        encoder.setBytes(&params, length: MemoryLayout<TemporalBoostParams>.stride,
                         index: 5)
        encoder.setBuffer(resources.changed, offset: 0, index: 6)
        encoder.setTexture(state.previousResidual, index: 0)
        encoder.setTexture(state.nextResidual, index: 1)
        dispatch2D(encoder, pso: resources.pso, w: roi.width, h: roi.height)
        encoder.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        guard command.error == nil else { return .unavailable }

        candidateBGR.withUnsafeMutableBytes {
            _ = memcpy($0.baseAddress!, resources.candidate.contents(), $0.count)
        }
        let oldPrevious = state.previousResidual
        state.previousResidual = state.nextResidual
        state.nextResidual = oldPrevious
        landmarks.withUnsafeBytes {
            _ = memcpy(state.previousLandmarks.contents(), $0.baseAddress!, $0.count)
        }
        state.previousPointValues = landmarks
        state.hasHistory = true
        state.lastFrameID = frameID
        let changed = resources.changed.contents()
            .assumingMemoryBound(to: UInt32.self).pointee != 0
        return useHistory && beta > 0 && changed ? .effect : .identity
    }

    /// Compatibility shim for the standalone parity harness. `true` means the
    /// frame was handled successfully, not that history changed its pixels.
    @discardableResult
    func temporalBoostFrame(candidateBGR: inout [UInt8], hostBGR: [UInt8],
                            support: [Float], landmarks: [SIMD2<Float>],
                            frameID: Int, reset: Bool,
                            beta: Float = Serve320MetalCompositor.temporalBoostBeta,
                            state: TemporalBoostState) -> Bool {
        applyTemporalBoostFrame(
            candidateBGR: &candidateBGR,
            hostBGR: hostBGR,
            support: support,
            landmarks: landmarks,
            frameID: frameID,
            reset: reset,
            beta: beta,
            state: state
        ).handled
    }

    private func ensureTemporalBoostResources() -> TemporalBoostResources? {
        if let temporalBoostResources { return temporalBoostResources }
        let roi = Self.temporalBoostROI
        let bytes = roi.width * roi.height * 3
        let supportBytes = roi.width * roi.height * MemoryLayout<Float>.stride
        guard let function = library.makeFunction(name: "serve320_temporal_boost"),
              let pso = try? device.makeComputePipelineState(function: function),
              let candidate = device.makeBuffer(length: bytes, options: .storageModeShared),
              let host = device.makeBuffer(length: bytes, options: .storageModeShared),
              let support = device.makeBuffer(length: supportBytes, options: .storageModeShared),
              let landmarks = device.makeBuffer(
                length: 20 * MemoryLayout<SIMD2<Float>>.stride,
                options: .storageModeShared),
              let changed = device.makeBuffer(
                length: MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else { return nil }
        let resources = TemporalBoostResources(pso: pso, candidate: candidate,
                                               host: host, support: support,
                                               currentLandmarks: landmarks,
                                               changed: changed)
        temporalBoostResources = resources
        return resources
    }

    // MARK: - per frame

    /// Produces the final opaque 1080x1920 canvas as a GPU-resident texture.
    /// `nil` means the optional ring is temporarily unavailable and the caller
    /// may use the established readback/CGImage path. The returned surface
    /// carries a shared-event value; consumers wait on the GPU without blocking
    /// this producer thread. Failures are reported through telemetry and force
    /// the event forward so a drawable cannot deadlock behind a lost command.
    func compositeFrameForPresentation(
        canvas: [UInt8], canvasW: Int, canvasH: Int,
        predBGR: [UInt8], support: [Float],
        box: (x0: Int, y0: Int, x1: Int, y1: Int),
        jawRestore: [JawRestorePixel],
        silence: SilenceApplication?
    ) throws -> PresentationSurface? {
        lastFrameTimings = nil
        guard Self.presentationFlagEnabled,
              let jawRestorePSO,
              let canvasToTexturePSO,
              canvasW == Self.presentationCanvasW,
              canvasH == Self.presentationCanvasH,
              let lease = acquirePresentationSlot() else {
            return nil
        }
        let (slot, resources, readyEvent, readyValue) = lease
        let texture = resources.texture
        var handedOff = false
        defer {
            if !handedOff { releasePresentationSlot(slot) }
        }

        let bw = box.x1 - box.x0, bh = box.y1 - box.y0
        let boxValid = bw > 0 && bh > 0 && box.x0 >= 0 && box.y0 >= 0
            && box.x1 <= canvasW && box.y1 <= canvasH
        guard canvas.count == canvasW * canvasH * 4,
              predBGR.count == res * res * 3,
              support.count == res * res else { return nil }
        if let silence, !configureSilence(for: silence.ema,
                                          canvasW: canvasW, canvasH: canvasH) {
            return nil
        }
        let jawBytes = jawRestore.count * MemoryLayout<JawRestorePixel>.stride
        if jawBytes > 0,
           (resources.jawRestore == nil
                || resources.jawRestore!.length < jawBytes) {
            resources.jawRestore = device.makeBuffer(
                length: jawBytes, options: .storageModeShared)
        }
        guard (jawBytes == 0 || resources.jawRestore != nil),
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return nil }
        let canvasBuf = resources.canvas
        let uploadStarted = CACurrentMediaTime()

        // The texture is the complete frame, so seed the full buffer once.
        // This replaces the later full-frame CGContext copy and CI upload.
        canvas.withUnsafeBytes {
            _ = memcpy(canvasBuf.contents(), $0.baseAddress!, $0.count)
        }
        if boxValid {
            predBGR.withUnsafeBytes {
                _ = memcpy(resources.prediction.contents(),
                           $0.baseAddress!, $0.count)
            }
            support.withUnsafeBytes {
                _ = memcpy(resources.support.contents(),
                           $0.baseAddress!, $0.count)
            }
            var params = PasteParams(canvasW: Int32(canvasW),
                                     boxX0: Int32(box.x0), boxY0: Int32(box.y0),
                                     boxW: Int32(bw), boxH: Int32(bh),
                                     srcRes: Int32(res),
                                     scaleX: Float(res) / Float(bw),
                                     scaleY: Float(res) / Float(bh))
            enc.setComputePipelineState(pastePSO)
            enc.setBuffer(canvasBuf, offset: 0, index: 0)
            enc.setBuffer(resources.prediction, offset: 0, index: 1)
            enc.setBuffer(resources.support, offset: 0, index: 2)
            enc.setBytes(&params, length: MemoryLayout<PasteParams>.stride, index: 3)
            dispatch2D(enc, pso: pastePSO, w: bw, h: bh)
        }
        if let silence {
            let step = silence.ema.stepMetal(frame: silence.frame,
                                             aperture: silence.aperture)
            encodeSilence(step, enc: enc, canvasBuf: canvasBuf, canvasW: canvasW)
        }
        if jawBytes > 0, let jawRestoreBuf = resources.jawRestore {
            jawRestore.withUnsafeBytes {
                _ = memcpy(jawRestoreBuf.contents(), $0.baseAddress!, $0.count)
            }
            var params = JawRestoreParams(count: UInt32(jawRestore.count))
            enc.setComputePipelineState(jawRestorePSO)
            enc.setBuffer(canvasBuf, offset: 0, index: 0)
            enc.setBuffer(jawRestoreBuf, offset: 0, index: 1)
            enc.setBytes(&params, length: MemoryLayout<JawRestoreParams>.stride, index: 2)
            dispatch1D(enc, pso: jawRestorePSO, count: jawRestore.count)
        }
        let uploadFinished = CACurrentMediaTime()
        var canvasParams = CanvasParams(canvasW: Int32(canvasW),
                                        canvasH: Int32(canvasH))
        enc.setComputePipelineState(canvasToTexturePSO)
        enc.setBuffer(canvasBuf, offset: 0, index: 0)
        enc.setTexture(texture, index: 0)
        enc.setBytes(&canvasParams, length: MemoryLayout<CanvasParams>.stride, index: 1)
        dispatch2D(enc, pso: canvasToTexturePSO, w: canvasW, h: canvasH)
        enc.endEncoding()
        cb.encodeSignalEvent(readyEvent, value: readyValue)
        cb.label = "Serve320 compositor frame \(readyValue)"
        let telemetry = presentationTelemetry
        cb.addCompletedHandler { buffer in
            if let error = buffer.error {
                // A failed producer must not strand the consumer queue behind
                // an event value that can never arrive. Advancing the event
                // releases the drawable; benchmark telemetry makes the frame
                // failure explicit instead of deadlocking the call.
                if readyEvent.signaledValue < readyValue {
                    readyEvent.signaledValue = readyValue
                }
                telemetry?(
                    "serve320_metal_present_failed",
                    "value=\(readyValue) error=\(error.localizedDescription)")
                return
            }
            let gpuMs = buffer.gpuEndTime > buffer.gpuStartTime
                ? (buffer.gpuEndTime - buffer.gpuStartTime) * 1_000.0 : 0
            telemetry?(
                "serve320_metal_present_gpu",
                String(format: "value=%llu gpu_ms=%.3f", readyValue, gpuMs))
        }
        cb.commit()
        let committed = CACurrentMediaTime()
        lastFrameTimings = FrameTimings(
            uploadMs: Self.milliseconds(uploadStarted, uploadFinished),
            encodeMs: Self.milliseconds(uploadFinished, committed),
            waitMs: 0,
            gpuMs: 0,
            readbackMs: 0)

        handedOff = true
        return PresentationSurface(
            texture: texture,
            readyEvent: readyEvent,
            readyValue: readyValue,
            producerCommandBuffer: cb,
            owner: self,
            slot: slot)
    }

    /// GPU replacement for `Serve320Compositor.canonicalCompositeNative` +
    /// `Serve320SilenceEMA.apply` on one frame. Returns false WITHOUT having
    /// consumed any silence-detector state, so the caller can run the CPU
    /// pair for this frame instead.
    func compositeFrame(canvas: inout [UInt8], canvasW: Int, canvasH: Int,
                        predBGR: [UInt8], support: [Float],
                        box: (x0: Int, y0: Int, x1: Int, y1: Int),
                        silence: SilenceApplication?) -> Bool {
        lastFrameTimings = nil
        let bw = box.x1 - box.x0, bh = box.y1 - box.y0
        // Same guard as canonicalCompositeNative — an invalid box pastes
        // nothing but the silence stack still runs.
        let boxValid = bw > 0 && bh > 0 && box.x0 >= 0 && box.y0 >= 0
            && box.x1 <= canvasW && box.y1 <= canvasH
        if !boxValid && silence == nil { return true }   // nothing to do (CPU ditto)
        guard canvas.count == canvasW * canvasH * 4,
              predBGR.count == res * res * 3,
              support.count == res * res else { return false }
        if let silence, !configureSilence(for: silence.ema,
                                          canvasW: canvasW, canvasH: canvasH) {
            return false
        }
        if canvasBuf == nil || canvasBuf!.length < canvas.count {
            canvasBuf = device.makeBuffer(length: canvas.count,
                                          options: .storageModeShared)
        }
        guard let canvasBuf,
              let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return false }
        let uploadStarted = CACurrentMediaTime()

        // Host -> GPU: only the row spans the kernels read or write.
        var spans: [(Int, Int)] = []
        if boxValid { spans.append((box.y0, box.y1)) }
        if silence != nil { spans.append((silenceRoi.y0, silenceRoi.y1)) }
        let rowBytes = canvasW * 4
        canvas.withUnsafeBytes { raw in
            for (y0, y1) in mergedSpans(spans) {
                memcpy(canvasBuf.contents() + y0 * rowBytes,
                       raw.baseAddress! + y0 * rowBytes, (y1 - y0) * rowBytes)
            }
        }
        if boxValid {
            predBGR.withUnsafeBytes { _ = memcpy(predBuf.contents(), $0.baseAddress!, $0.count) }
            support.withUnsafeBytes { _ = memcpy(supportBuf.contents(), $0.baseAddress!, $0.count) }
        }
        let uploadFinished = CACurrentMediaTime()
        if boxValid {
            var params = PasteParams(canvasW: Int32(canvasW),
                                     boxX0: Int32(box.x0), boxY0: Int32(box.y0),
                                     boxW: Int32(bw), boxH: Int32(bh),
                                     srcRes: Int32(res),
                                     scaleX: Float(res) / Float(bw),
                                     scaleY: Float(res) / Float(bh))
            enc.setComputePipelineState(pastePSO)
            enc.setBuffer(canvasBuf, offset: 0, index: 0)
            enc.setBuffer(predBuf, offset: 0, index: 1)
            enc.setBuffer(supportBuf, offset: 0, index: 2)
            enc.setBytes(&params, length: MemoryLayout<PasteParams>.stride, index: 3)
            dispatch2D(enc, pso: pastePSO, w: bw, h: bh)
        }
        if let silence {
            // Decision state is consumed only after every fallible make above
            // succeeded — a false return can no longer skip a detector frame.
            let step = silence.ema.stepMetal(frame: silence.frame,
                                             aperture: silence.aperture)
            encodeSilence(step, enc: enc, canvasBuf: canvasBuf, canvasW: canvasW)
        }
        enc.endEncoding()
        cb.commit()
        let committed = CACurrentMediaTime()
        cb.waitUntilCompleted()
        let completed = CACurrentMediaTime()
        let gpuMs = cb.gpuEndTime > cb.gpuStartTime
            ? (cb.gpuEndTime - cb.gpuStartTime) * 1_000.0 : 0
        if let error = cb.error {
            lastFrameTimings = FrameTimings(
                uploadMs: Self.milliseconds(uploadStarted, uploadFinished),
                encodeMs: Self.milliseconds(uploadFinished, committed),
                waitMs: Self.milliseconds(committed, completed),
                gpuMs: gpuMs,
                readbackMs: 0)
            // Post-commit GPU loss (device reset): the frame keeps its
            // pre-composite bytes; state was consumed, so do NOT fall back.
            print("[Serve320Metal] command buffer error: \(error)")
            return true
        }
        canvas.withUnsafeMutableBytes { raw in
            for (y0, y1) in mergedSpans(spans) {
                memcpy(raw.baseAddress! + y0 * rowBytes,
                       canvasBuf.contents() + y0 * rowBytes, (y1 - y0) * rowBytes)
            }
        }
        let readbackFinished = CACurrentMediaTime()
        lastFrameTimings = FrameTimings(
            uploadMs: Self.milliseconds(uploadStarted, uploadFinished),
            encodeMs: Self.milliseconds(uploadFinished, committed),
            waitMs: Self.milliseconds(committed, completed),
            gpuMs: gpuMs,
            readbackMs: Self.milliseconds(completed, readbackFinished))
        return true
    }

    // MARK: - silence ROI plumbing

    /// Kernel dispatch order mirrors the CPU statement order in
    /// Serve320SilenceEMA.apply: reg capture (replay.py:580), EMA write
    /// (:582-584), byte-freeze snapshot of the post-EMA canvas (:585-589),
    /// locked blend (:590), or disengage reset (:591-594). The serial compute
    /// encoder guarantees each dispatch sees the previous one's writes.
    private func encodeSilence(_ step: Serve320SilenceEMA.MetalStep,
                               enc: MTLComputeCommandEncoder,
                               canvasBuf: MTLBuffer, canvasW: Int) {
        guard let maskBuf, let regBuf, let emaBuf, let lockBuf else { return }
        let ww = silenceRoi.x1 - silenceRoi.x0
        let hh = silenceRoi.y1 - silenceRoi.y0
        var params = RoiParams(canvasW: Int32(canvasW),
                               roiX0: Int32(silenceRoi.x0), roiY0: Int32(silenceRoi.y0),
                               roiW: Int32(ww), roiH: Int32(hh),
                               beta: Serve320SilenceEMA.beta)
        let paramsLen = MemoryLayout<RoiParams>.stride

        enc.setComputePipelineState(roiCopyPSO)
        enc.setBuffer(canvasBuf, offset: 0, index: 0)
        enc.setBuffer(regBuf, offset: 0, index: 1)
        enc.setBytes(&params, length: paramsLen, index: 2)
        dispatch2D(enc, pso: roiCopyPSO, w: ww, h: hh)

        if step.engaged {
            enc.setComputePipelineState(emaPSO)
            enc.setBuffer(canvasBuf, offset: 0, index: 0)
            enc.setBuffer(regBuf, offset: 0, index: 1)
            enc.setBuffer(emaBuf, offset: 0, index: 2)
            enc.setBuffer(maskBuf, offset: 0, index: 3)
            enc.setBytes(&params, length: paramsLen, index: 4)
            dispatch2D(enc, pso: emaPSO, w: ww, h: hh)
            if step.takeLockSnapshot {
                enc.setComputePipelineState(roiCopyPSO)
                enc.setBuffer(canvasBuf, offset: 0, index: 0)
                enc.setBuffer(lockBuf, offset: 0, index: 1)
                enc.setBytes(&params, length: paramsLen, index: 2)
                dispatch2D(enc, pso: roiCopyPSO, w: ww, h: hh)
            }
            if step.blendLock {
                enc.setComputePipelineState(lockBlendPSO)
                enc.setBuffer(canvasBuf, offset: 0, index: 0)
                enc.setBuffer(regBuf, offset: 0, index: 1)
                enc.setBuffer(lockBuf, offset: 0, index: 2)
                enc.setBuffer(maskBuf, offset: 0, index: 3)
                enc.setBytes(&params, length: paramsLen, index: 4)
                dispatch2D(enc, pso: lockBlendPSO, w: ww, h: hh)
            }
        } else if step.resetState {
            enc.setComputePipelineState(emaResetPSO)
            enc.setBuffer(regBuf, offset: 0, index: 0)
            enc.setBuffer(emaBuf, offset: 0, index: 1)
            enc.setBytes(&params, length: paramsLen, index: 2)
            dispatch2D(enc, pso: emaResetPSO, w: ww, h: hh)
        }
    }

    private func configureSilence(for ema: Serve320SilenceEMA,
                                  canvasW: Int, canvasH: Int) -> Bool {
        if silenceOwner === ema { return true }
        let roi = ema.roi
        let ww = roi.x1 - roi.x0, hh = roi.y1 - roi.y0
        guard ww > 0, hh > 0, roi.x0 >= 0, roi.y0 >= 0,
              roi.x1 <= canvasW, roi.y1 <= canvasH,
              ema.mask.count == ww * hh else { return false }
        guard let mask = ema.mask.withUnsafeBytes({ raw in
                  device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                                    options: .storageModeShared)
              }),
              let reg = device.makeBuffer(length: ww * hh * 4,
                                          options: .storageModeShared),
              let emaState = device.makeBuffer(length: ww * hh * 4 * MemoryLayout<Float>.stride,
                                               options: .storageModeShared),
              let lock = device.makeBuffer(length: ww * hh * 4,
                                           options: .storageModeShared) else {
            return false
        }
        // Fresh owner ⇒ fresh state machine: stepMetal's first frame always
        // resets the EMA from live pixels and a lock snapshot always precedes
        // a lock blend, so no stale buffer content can ever be read.
        silenceOwner = ema
        silenceRoi = (roi.x0, roi.y0, roi.x1, roi.y1)
        maskBuf = mask
        regBuf = reg
        emaBuf = emaState
        lockBuf = lock
        return true
    }

    // MARK: - helpers

    private func dispatch2D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState, w: Int, h: Int) {
        let tw = pso.threadExecutionWidth
        let th = max(pso.maxTotalThreadsPerThreadgroup / tw, 1)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    }

    private static func milliseconds(_ start: CFTimeInterval,
                                     _ end: CFTimeInterval) -> Double {
        max(0, end - start) * 1_000.0
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder,
                            pso: MTLComputePipelineState, count: Int) {
        let width = min(pso.maxTotalThreadsPerThreadgroup,
                        max(pso.threadExecutionWidth, 1))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }

    private func acquirePresentationSlot()
        -> (Int, PresentationSlot, MTLSharedEvent, UInt64)? {
        presentationLock.lock()
        defer { presentationLock.unlock() }
        guard let presentationEvent,
              presentationSlots.count == presentationLeased.count,
              nextPresentationSignalValue < UInt64.max else { return nil }
        for offset in 0..<presentationLeased.count {
            let slot = (nextPresentationSlot + offset) % presentationLeased.count
            if !presentationLeased[slot] {
                presentationLeased[slot] = true
                nextPresentationSlot = (slot + 1) % presentationLeased.count
                let signalValue = nextPresentationSignalValue
                nextPresentationSignalValue += 1
                return (slot, presentationSlots[slot],
                        presentationEvent, signalValue)
            }
        }
        return nil
    }

    private func releasePresentationSlot(_ slot: Int) {
        presentationLock.lock()
        if presentationLeased.indices.contains(slot) {
            presentationLeased[slot] = false
        }
        presentationLock.unlock()
    }

    /// Merge possibly-overlapping row spans so each byte is copied once.
    private func mergedSpans(_ spans: [(Int, Int)]) -> [(Int, Int)] {
        let sorted = spans.filter { $0.1 > $0.0 }.sorted { $0.0 < $1.0 }
        var out: [(Int, Int)] = []
        for s in sorted {
            if var last = out.last, s.0 <= last.1 {
                last.1 = max(last.1, s.1)
                out[out.count - 1] = last
            } else {
                out.append(s)
            }
        }
        return out
    }
}

/// One temporal runtime per assistant turn stream. TextChain creates many
/// pipelines for one streamed reply, so putting this state on a pipeline loses
/// history at every chunk boundary. A lease also prevents a canceled pipeline
/// from repopulating state after a confirmed interruption reset.
final class Serve320TemporalBoostSession: @unchecked Sendable {
    typealias Telemetry = @Sendable (_ event: String, _ detail: String) -> Void

    private let lock = NSLock()
    private var lifecycle = Serve320TemporalBoostLifecycle()
    private var runtime: Serve320MetalCompositor?
    private var state: Serve320MetalCompositor.TemporalBoostState?
    private let telemetry: Telemetry

    init(telemetry: @escaping Telemetry = { _, _ in }) {
        self.telemetry = telemetry
    }

    var currentLease: Serve320TemporalBoostLease {
        lock.lock()
        defer { lock.unlock() }
        return lifecycle.currentLease
    }

    /// Reset recurrent history and invalidate every pipeline created before
    /// this boundary. The Metal initialization attempt remains negative-cached.
    @discardableResult
    func reset(reason: String) -> Serve320TemporalBoostLease {
        lock.lock()
        let lease = lifecycle.reset()
        state?.reset()
        lock.unlock()
        telemetry("serve320_temporal_boost_reset", "reason=\(reason)")
        return lease
    }

    func apply(candidateBGR: inout [UInt8], hostBGR: [UInt8],
               support: [Float], landmarks: [SIMD2<Float>],
               frameID: Int, lease: Serve320TemporalBoostLease)
        -> Serve320TemporalBoostApplication {
        lock.lock()
        defer { lock.unlock() }
        guard lifecycle.accepts(lease) else { return .unavailable }

        if runtime == nil || state == nil {
            guard lifecycle.beginInitializationAttempt() else {
                return .unavailable
            }
            guard let source = Serve320MetalCompositor.loadBundledShaderSource(),
                  let newRuntime = Serve320MetalCompositor(shaderSource: source),
                  let newState = newRuntime.makeTemporalBoostState() else {
                telemetry(
                    "serve320_temporal_boost_unavailable",
                    "reason=metal_initialization_failed"
                )
                return .unavailable
            }
            runtime = newRuntime
            state = newState
            telemetry("serve320_temporal_boost_ready", "")
        }

        guard let runtime, let state else { return .unavailable }
        return runtime.applyTemporalBoostFrame(
            candidateBGR: &candidateBGR,
            hostBGR: hostBGR,
            support: support,
            landmarks: landmarks,
            frameID: frameID,
            reset: false,
            state: state
        )
    }
}
