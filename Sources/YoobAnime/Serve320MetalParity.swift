//
//  Serve320MetalParity.swift
//  CPU-vs-Metal comparator for the Serve320 composite kernels. The CPU pair
//  (Serve320Compositor.canonicalCompositeNative + Serve320SilenceEMA.apply)
//  is the parity truth; the Metal path must match every canvas byte to
//  max abs diff <= 1 on random frames (expected 0 — the <=1 headroom exists
//  only for possible FMA-contraction ulps at exact-.5 rounding boundaries,
//  see Compositor.metal header).
//
//  Two callers share this code so "what was tested" == "what ships":
//    - scripts/serve320_metal_parity.swift (standalone, full suite + timing)
//    - Serve320MetalCompositor.createVerifiedForProduct (on-device quick
//      self-check at startup of the flagged mode; pass persisted in
//      UserDefaults).
//
//  Coverage per config: upsample/downsample/identity boxes (identity boxes
//  hit exact-.5 blend inputs, proving half-even rounding, not just half-up),
//  sup==0 skip, EMA engage -> converge -> hard-lock snapshot -> locked blend
//  -> instant release -> re-engage, and the audit state machines compared
//  row-by-row between the CPU and Metal silence instances.
//

import Foundation

enum Serve320MetalParity {

    struct Report {
        var pass: Bool
        var maxAbsDiff: Int
        var framesCompared: Int
        var mismatchedBytes: Int
        var detail: String
    }

    /// Deterministic LCG, same recipe as scripts/serve320_imageops_parity.swift.
    struct LCG {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
        mutating func byte() -> UInt8 { UInt8((next() >> 40) & 0xFF) }
        mutating func float01() -> Float { Float((next() >> 40) & 0xFFFF) / Float(0xFFFF) }
        mutating func int(_ range: Range<Int>) -> Int {
            range.lowerBound + Int(next() % UInt64(range.count))
        }
        mutating func bytes(_ n: Int) -> [UInt8] {
            var out = [UInt8](repeating: 0, count: n)
            let words = (n + 7) / 8
            out.withUnsafeMutableBytes { raw in
                for w in 0..<words {
                    var v = next()
                    let base = w * 8
                    let len = Swift.min(8, n - base)
                    memcpy(raw.baseAddress! + base, &v, len)
                }
            }
            return out
        }
    }

    private struct Config {
        let name: String
        let canvasW: Int
        let canvasH: Int
        let frames: Int
        let silence: Bool
        let hardLock: Bool
        let freezeCanvas: Bool      // constant canvas across frames (EMA converges)
        let rawBox: [Int32]         // single host row -> ROI (canvas coords)
        let seed: UInt64
    }

    // Aperture schedule exercising every silence branch: 3 warmup frames
    // (hold not yet satisfied), engage at 3, hard-lock run reaches 8 at
    // frame 10 (snapshot + locked blends 10..14), instant release at 15
    // (open mouth), history-poisoned frames 16..18, re-engage at 19.
    private static func apertureSchedule(_ frames: Int) -> [Float] {
        (0..<frames).map { i in
            switch i {
            case 0..<3: return 0.50
            case 15: return 0.80
            default: return 0.10
            }
        }
    }

    private static func configs(quick: Bool) -> [Config] {
        var cfgs: [Config] = [
            // Identity box (320x320, scale 1): taps collapse to the source
            // pixel, support carries exact .5s -> byte-exact only when the
            // Metal blend rounds half-even like np.rint.
            Config(name: "paste-identity-halfeven", canvasW: 480, canvasH: 854,
                   frames: 4, silence: false, hardLock: false, freezeCanvas: false,
                   rawBox: [], seed: 0x320_2026_0001),
            // Odd-sized up/downsample boxes, random support with zero patches.
            Config(name: "paste-resample", canvasW: 480, canvasH: 854,
                   frames: quick ? 5 : 10, silence: false, hardLock: false,
                   freezeCanvas: false, rawBox: [], seed: 0x320_2026_0002),
            // Full silence stack, hard-lock ON, frozen canvas (EMA converges,
            // snapshot then byte-frozen core must reproduce verbatim).
            Config(name: "silence-hardlock", canvasW: 480, canvasH: 854,
                   frames: 24, silence: true, hardLock: true, freezeCanvas: true,
                   rawBox: [140, 420, 340, 560], seed: 0x320_2026_0003),
            // Silence EMA only, moving canvas (fresh random bytes per frame).
            Config(name: "silence-ema-moving", canvasW: 480, canvasH: 854,
                   frames: quick ? 12 : 24, silence: true, hardLock: false,
                   freezeCanvas: false, rawBox: [150, 430, 330, 540],
                   seed: 0x320_2026_0004),
        ]
        if !quick {
            cfgs.append(Config(name: "full-canvas-hardlock", canvasW: 1080, canvasH: 1920,
                               frames: 24, silence: true, hardLock: true,
                               freezeCanvas: false, rawBox: [400, 700, 680, 900],
                               seed: 0x320_2026_0005))
        }
        return cfgs
    }

    /// Boxes cycle through upsample (typical serve stab box ~1.9x), odd-size
    /// upsample, identity, and downsample; all inside the canvas.
    private static func box(for cfg: Config, frame: Int, rng: inout LCG)
        -> (x0: Int, y0: Int, x1: Int, y1: Int) {
        if cfg.name == "paste-identity-halfeven" {
            let x0 = rng.int(0..<(cfg.canvasW - 320))
            let y0 = rng.int(0..<(cfg.canvasH - 320))
            return (x0, y0, x0 + 320, y0 + 320)
        }
        let shapes: [(Int, Int)] = [(457, 521), (320, 320), (200, 180),
                                    (cfg.canvasW >= 1080 ? 608 : 440,
                                     cfg.canvasH >= 1920 ? 608 : 440)]
        let (bw, bh) = shapes[frame % shapes.count]
        let x0 = rng.int(0..<max(cfg.canvasW - bw, 1))
        let y0 = rng.int(0..<max(cfg.canvasH - bh, 1))
        return (x0, y0, x0 + bw, y0 + bh)
    }

    private static func randomSupport(for cfg: Config, rng: inout LCG) -> [Float] {
        let plane = 320 * 320
        var sup = [Float](repeating: 0, count: plane)
        if cfg.name == "paste-identity-halfeven" {
            // Exact-.5 alpha everywhere: region/pred integer parity decides
            // whether the blend lands on n+.5, the half-even litmus.
            for i in 0..<plane { sup[i] = 0.5 }
            // Sprinkle exact 0/1 to keep the skip and passthrough lanes hot.
            for _ in 0..<2000 { sup[rng.int(0..<plane)] = 0 }
            for _ in 0..<2000 { sup[rng.int(0..<plane)] = 1 }
            return sup
        }
        for i in 0..<plane {
            let r = rng.float01()
            if r < 0.30 { sup[i] = 0 }            // outside-support skip
            else if r < 0.38 { sup[i] = 1 }       // hard interior
            else { sup[i] = rng.float01() }       // feather
        }
        return sup
    }

    /// Run the comparator. `quick` is the on-device startup self-check
    /// (smaller canvases, fewer frames, < ~1 s); the standalone harness runs
    /// the full suite including a 1080x1920 config.
    static func run(metal: Serve320MetalCompositor, quick: Bool) -> Report {
        var maxDiff = 0
        var mismatched = 0
        var frames = 0
        var lines: [String] = []
        for cfg in configs(quick: quick) {
            var rng = LCG(state: cfg.seed)
            let canvasBytes = cfg.canvasW * cfg.canvasH * 4
            let aps = apertureSchedule(cfg.frames)
            let rms = [Float](repeating: 0, count: cfg.frames)
            let cpuEMA = cfg.silence
                ? Serve320SilenceEMA(rawBoxes: cfg.rawBox, frameCount: cfg.frames,
                                     rms: rms, hardLock: cfg.hardLock) : nil
            let gpuEMA = cfg.silence
                ? Serve320SilenceEMA(rawBoxes: cfg.rawBox, frameCount: cfg.frames,
                                     rms: rms, hardLock: cfg.hardLock) : nil
            let frozen = cfg.freezeCanvas ? rng.bytes(canvasBytes) : []
            var cfgMax = 0
            for i in 0..<cfg.frames {
                let canvas = cfg.freezeCanvas ? frozen : rng.bytes(canvasBytes)
                let pred = rng.bytes(320 * 320 * 3)
                let support = randomSupport(for: cfg, rng: &rng)
                let b = box(for: cfg, frame: i, rng: &rng)

                var cpuCanvas = canvas
                Serve320Compositor.canonicalCompositeNative(
                    canvas: &cpuCanvas, canvasW: cfg.canvasW, canvasH: cfg.canvasH,
                    predBGR: pred, support: support, box: b)
                cpuEMA?.apply(frame: i, aperture: aps[i],
                              canvas: &cpuCanvas, canvasW: cfg.canvasW)

                var gpuCanvas = canvas
                let ok = metal.compositeFrame(
                    canvas: &gpuCanvas, canvasW: cfg.canvasW, canvasH: cfg.canvasH,
                    predBGR: pred, support: support, box: b,
                    silence: gpuEMA.map {
                        Serve320MetalCompositor.SilenceApplication(
                            ema: $0, frame: i, aperture: aps[i])
                    })
                guard ok else {
                    return Report(pass: false, maxAbsDiff: 255, framesCompared: frames,
                                  mismatchedBytes: mismatched,
                                  detail: "\(cfg.name) frame \(i): compositeFrame returned false")
                }
                frames += 1
                guard cpuCanvas != gpuCanvas else { continue }   // memcmp fast path
                for j in 0..<canvasBytes {
                    let d = abs(Int(cpuCanvas[j]) - Int(gpuCanvas[j]))
                    if d > 0 {
                        mismatched += 1
                        if d > cfgMax { cfgMax = d }
                    }
                }
            }
            if let cpuEMA, let gpuEMA {
                let same = cpuEMA.audit.count == gpuEMA.audit.count
                    && zip(cpuEMA.audit, gpuEMA.audit).allSatisfy {
                        $0.silent == $1.silent && $0.engaged == $1.engaged
                            && $0.locked == $1.locked
                    }
                if !same {
                    return Report(pass: false, maxAbsDiff: max(maxDiff, cfgMax),
                                  framesCompared: frames, mismatchedBytes: mismatched,
                                  detail: "\(cfg.name): CPU/Metal silence audit sequences diverge")
                }
            }
            maxDiff = max(maxDiff, cfgMax)
            lines.append("\(cfg.name): \(cfg.frames)f max|d|=\(cfgMax)")
        }
        let pass = maxDiff <= 1
        return Report(pass: pass, maxAbsDiff: maxDiff, framesCompared: frames,
                      mismatchedBytes: mismatched,
                      detail: lines.joined(separator: "; ")
                          + " | mismatched bytes: \(mismatched)")
    }

    /// CPU-vs-Metal wall-clock on realistic dims (standalone harness only).
    /// Returns (cpuMsPerFrame, metalMsPerFrame) incl. host<->GPU transfers.
    static func benchmark(metal: Serve320MetalCompositor,
                          frames: Int = 50) -> (cpuMs: Double, metalMs: Double) {
        var rng = LCG(state: 0x320_2026_00BE)
        let cw = 1080, ch = 1920
        let box = (x0: 236, y0: 690, x1: 236 + 608, y1: 690 + 608)
        let rawBox: [Int32] = [400, 700, 680, 900]
        let rms = [Float](repeating: 0, count: frames)
        let aps = apertureSchedule(frames)
        let canvas = rng.bytes(cw * ch * 4)
        let pred = rng.bytes(320 * 320 * 3)
        let support = (0..<320 * 320).map { _ in rng.float01() }
        let cpuEMA = Serve320SilenceEMA(rawBoxes: rawBox, frameCount: frames,
                                        rms: rms, hardLock: true)
        let gpuEMA = Serve320SilenceEMA(rawBoxes: rawBox, frameCount: frames,
                                        rms: rms, hardLock: true)

        var cpuCanvas = canvas
        let t0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<frames {
            cpuCanvas = canvas
            Serve320Compositor.canonicalCompositeNative(
                canvas: &cpuCanvas, canvasW: cw, canvasH: ch,
                predBGR: pred, support: support, box: box)
            cpuEMA.apply(frame: i, aperture: aps[i], canvas: &cpuCanvas, canvasW: cw)
        }
        let t1 = DispatchTime.now().uptimeNanoseconds
        var gpuCanvas = canvas
        for i in 0..<frames {
            gpuCanvas = canvas
            _ = metal.compositeFrame(
                canvas: &gpuCanvas, canvasW: cw, canvasH: ch,
                predBGR: pred, support: support, box: box,
                silence: Serve320MetalCompositor.SilenceApplication(
                    ema: gpuEMA, frame: i, aperture: aps[i]))
        }
        let t2 = DispatchTime.now().uptimeNanoseconds
        return (Double(t1 - t0) / 1e6 / Double(frames),
                Double(t2 - t1) / 1e6 / Double(frames))
    }
}
