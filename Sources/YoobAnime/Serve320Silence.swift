//
//  Serve320Silence.swift
//  QA13 output-ROI silence EMA + QA15 silence hard-lock for the Serve320 lane,
//  ported from arm_a/replay.py (b200 serve). One shared measured-frozen-geometry
//  detector is the single source of truth for "motion" (replay.py:502-505):
//    silent = i >= 3 AND ap_i < 0.35 AND |ap[i-1-k] - ap_i| < 0.01 for k in 0..2
//             AND aud_energy[i] < 0.06   (aud_energy = per-frame RMS / p90 of clip)
//
//  QA13 (replay.py:466-477 setup, 578-584 apply): during silent frames, EMA the
//  COMPOSITED full-frame mouth ROI toward stillness (beta 0.75) through a
//  feathered ramp mask; HARD disengage on the first motion frame (speech frames
//  byte-identical). ROI = union of the used host frames' RAW boxes + 15% pad.
//  QA15 (replay.py:58-66 _hard_lock_step + 585-594): after 8 consecutive engaged
//  frames, BYTE-FREEZE the converged ROI (verbatim in the mask core, feather
//  live); INSTANT release on the first non-silent frame; fresh hold to re-lock.
//
//  Defaults: EMA ON for the product serve paths (canned sample + live text),
//  hard-lock OFF unless AVATAR_SERVE320_HARD_LOCK=1 (owner wants zero silence
//  motion but reviewers flagged statue-risk — ship gated, measure first).
//

import Foundation

final class Serve320SilenceEMA {
    struct ROI { let y0: Int, y1: Int, x0: Int, x1: Int }

    let roi: ROI
    let mask: [Float]          // (hh*ww) ramp, replay.py:473-477 (1 core -> 0 edge)
    let energy: [Float]        // replay.py:465-467 _aud_energy (rms / p90)
    let hardLockEnabled: Bool

    static let beta: Float = 0.75         // replay.py:184 out_ema_beta
    static let hold = 3                   // replay.py:174 ema_silence_hold
    static let eps: Float = 0.01          // replay.py:174 ema_silence_eps
    static let silenceFrac: Float = 0.06  // replay.py:186 audio_silence_frac
    static let hardLockHold = 8           // replay.py:192 hard_lock_hold
    static let hostMotionEps: Float = 1.25  // px/frame of raw-box center drift

    private var apertures: [Float] = []
    private var lastFrame: Int?
    /// Per-host-row raw-box center drift (px vs previous row, wrap included).
    /// QA13 was validated on a near-still hold span; the blink-restored spans
    /// MOVE, and an EMA over a moving host lags the ROI interior behind the
    /// live surround — the visible rectangle + ghosted double lips. The EMA
    /// may only engage while the host itself is quiet.
    private let rowMotion: [Float]
    private let rowCount: Int
    private var emaState: [Float]?        // hh*ww*4 f32 (BGRA; alpha passes through)
    private var lockState: [UInt8]?       // hh*ww*4 u8 snapshot of the converged ROI
    private var lockRun = 0

    struct AuditFrame {
        var aperture: Float
        var energy: Float
        var silent: Bool
        var engaged: Bool
        var locked: Bool
    }
    private(set) var audit: [AuditFrame] = []

    /// rawBoxes: (284,4) i32 raw mouth boxes of the idle span (replay.py:469
    /// uses the K160 raw_box of the hosts used; hosts = idle frames [0, min(n,284))).
    /// - Parameter energyReference: p90 of the RMS the energy gate normalises
    ///   against. Defaults to the p90 of `rms` itself, which is right for a
    ///   whole utterance. A streaming payload is as short as 0.6 s, and a
    ///   payload that happens to be all pause would otherwise rescale its own
    ///   quiet to 1.0 and never register as silent; the streaming caller passes
    ///   the p90 of the wider model window instead.
    init(rawBoxes: [Int32], frameCount: Int, rms: [Float], hardLock: Bool,
         energyReference: Float? = nil) {
        let use = min(frameCount, rawBoxes.count / 4)
        let rows = rawBoxes.count / 4
        rowCount = rows
        var motion = [Float](repeating: 0, count: max(rows, 1))
        if rows > 1 {
            for r in 0..<rows {
                let p = (r + rows - 1) % rows   // wrap: the loop seam counts as motion
                let cx = Float(rawBoxes[r * 4] + rawBoxes[r * 4 + 2]) * 0.5
                let cy = Float(rawBoxes[r * 4 + 1] + rawBoxes[r * 4 + 3]) * 0.5
                let px = Float(rawBoxes[p * 4] + rawBoxes[p * 4 + 2]) * 0.5
                let py = Float(rawBoxes[p * 4 + 1] + rawBoxes[p * 4 + 3]) * 0.5
                motion[r] = max(abs(cx - px), abs(cy - py))
            }
        }
        rowMotion = motion
        var mx0 = Int32.max, my0 = Int32.max, mx1 = Int32.min, my1 = Int32.min
        for i in 0..<use {
            mx0 = min(mx0, rawBoxes[i * 4]); my0 = min(my0, rawBoxes[i * 4 + 1])
            mx1 = max(mx1, rawBoxes[i * 4 + 2]); my1 = max(my1, rawBoxes[i * 4 + 3])
        }
        // replay.py:470-472: +15% pad of the max span
        let pad = Int(0.15 * Float(max(mx1 - mx0, my1 - my0)))
        roi = ROI(y0: max(0, Int(my0) - pad), y1: Int(my1) + pad,
                  x0: max(0, Int(mx0) - pad), x1: Int(mx1) + pad)
        let hh = roi.y1 - roi.y0, ww = roi.x1 - roi.x0
        // replay.py:474-477: ramp = min(arange, arange[::-1])/fe per axis, clip
        let fe = Float(max(8, Int(0.12 * Double(min(hh, ww)))))
        var m = [Float](repeating: 0, count: hh * ww)
        for y in 0..<hh {
            for x in 0..<ww {
                let dx = min(x, ww - 1 - x)
                let dy = min(y, hh - 1 - y)
                m[y * ww + x] = min(max(Float(min(dx, dy)) / fe, 0), 1)
            }
        }
        mask = m
        let p90 = energyReference ?? Self.percentile90(rms)
        energy = rms.map { $0 / (p90 + 1e-6) }
        hardLockEnabled = hardLock
    }

    /// np.percentile(_e, 90) — linear interpolation (numpy default method).
    static func percentile90(_ values: [Float]) -> Float {
        let s = values.sorted()
        guard !s.isEmpty else { return 0 }
        let pos = 0.9 * Double(s.count - 1)
        let lo = Int(pos.rounded(.down))
        let frac = Float(pos - Double(lo))
        let hi = min(lo + 1, s.count - 1)
        return s[lo] * (1 - frac) + s[hi] * frac
    }

    static var hardLockDefault: Bool {
        ProcessInfo.processInfo.environment["AVATAR_SERVE320_HARD_LOCK"] == "1"
    }

    /// The measured-frozen-geometry + audio + host-motion silence detector,
    /// shared verbatim by the CPU pixel path (`apply`) and the Metal decision
    /// mirror (`stepMetal`). Mutates the aperture history / gap tracking.
    private func detectSilence(frame i: Int, aperture ap: Float) -> Bool {
        let contiguous = lastFrame.map { i == $0 + 1 } ?? (i == 0)
        if !contiguous {
            // A skip or a new utterance starts a fresh hold window. Keeping
            // pre-gap samples would let the second post-gap frame compare
            // against old history and re-engage the silence EMA too early.
            apertures.removeAll(keepingCapacity: true)
        }
        lastFrame = i
        apertures.append(ap)
        let apertureIndex = apertures.count - 1
        // replay.py:502-505 — the measured-frozen-geometry + audio-gate detector
        // A realtime player may skip an expired video frame to stay on the
        // audio clock. A gap resets the silence hold instead of indexing the
        // compact rendered-frame history with the original feature index.
        var silent = contiguous && apertureIndex >= Self.hold && ap < 0.35
        if silent {
            for k in 0..<Self.hold {
                if abs(apertures[apertureIndex - 1 - k] - ap) >= Self.eps {
                    silent = false
                    break
                }
            }
        }
        if silent, energy[i] >= Self.silenceFrac { silent = false }
        // Host-motion gate (device-only guard, see rowMotion above): the host
        // row for output frame i is i % nIdle (Serve320Pipeline host_loop).
        if silent, rowCount > 1 {
            let hostRow = i % rowCount
            for k in 0...Self.hold where silent {
                let r = (hostRow + rowCount - (k % rowCount)) % rowCount
                if rowMotion[r] >= Self.hostMotionEps { silent = false }
            }
        }
        return silent
    }

    /// Per-frame apply (replay.py:578-596). `ap` = this frame's predicted aperture;
    /// `canvas` = composited BGRA bytes (1080x1920*4), mutated in the ROI only.
    func apply(frame i: Int, aperture ap: Float, canvas: inout [UInt8], canvasW: Int) {
        let silent = detectSilence(frame: i, aperture: ap)

        let hh = roi.y1 - roi.y0, ww = roi.x1 - roi.x0
        // reg = current composited ROI (replay.py:580)
        var reg = [Float](repeating: 0, count: hh * ww * 4)
        for y in 0..<hh {
            let srcRow = ((roi.y0 + y) * canvasW + roi.x0) * 4
            for x in 0..<ww {
                for c in 0..<4 {
                    reg[(y * ww + x) * 4 + c] = Float(canvas[srcRow + x * 4 + c])
                }
            }
        }

        var engaged = false
        if silent, emaState != nil {
            engaged = true
            // replay.py:582-584 — EMA toward stillness, masked write (np.rint half-even)
            var ema = emaState!
            for p in 0..<(hh * ww) {
                let m = mask[p]
                for c in 0..<3 {   // BGR only; alpha passes through untouched
                    let idx = p * 4 + c
                    ema[idx] = Self.beta * ema[idx] + (1 - Self.beta) * reg[idx]
                    let v = (reg[idx] * (1 - m) + ema[idx] * m).rounded(.toNearestOrEven)
                    canvas[(roi.y0 + p / ww) * canvasW * 4 + (roi.x0 + p % ww) * 4 + c] =
                        UInt8(min(max(v, 0), 255))
                }
            }
            emaState = ema
            if hardLockEnabled {
                // replay.py:58-66 _hard_lock_step(True) + 585-594
                lockRun += 1
                let locked = lockRun >= Self.hardLockHold
                if locked {
                    if lockState == nil {
                        // snapshot the converged ROI (post-EMA write)
                        var snap = [UInt8](repeating: 255, count: hh * ww * 4)
                        for y in 0..<hh {
                            let srcRow = ((roi.y0 + y) * canvasW + roi.x0) * 4
                            for x in 0..<ww * 4 {
                                snap[y * ww * 4 + x] = canvas[srcRow + x]
                            }
                        }
                        lockState = snap
                    }
                    if let snap = lockState {
                        // replay.py:590 — comp = rint(reg*(1-m) + lock*m): frozen core, live feather
                        for p in 0..<(hh * ww) {
                            let m = mask[p]
                            for c in 0..<3 {
                                let idx = p * 4 + c
                                let v = (reg[idx] * (1 - m) + Float(snap[idx]) * m).rounded(.toNearestOrEven)
                                canvas[(roi.y0 + p / ww) * canvasW * 4 + (roi.x0 + p % ww) * 4 + c] =
                                    UInt8(min(max(v, 0), 255))
                            }
                        }
                    }
                }
            }
        } else {
            // replay.py:591-594 — disengage: track the live frame; INSTANT lock release
            emaState = reg
            if hardLockEnabled {
                lockRun = 0
                lockState = nil
            }
        }
        audit.append(AuditFrame(aperture: ap, energy: energy[i], silent: silent,
                                engaged: engaged, locked: lockState != nil))
    }

    // MARK: - Metal-path decision mirror (AVATAR_METAL_COMPOSITE=1)

    /// What Serve320MetalCompositor must dispatch for one frame. The pixel
    /// state (ema f32, lock snapshot bytes) lives in GPU buffers; this struct
    /// carries only the branch the CPU reference would have taken.
    struct MetalStep {
        let engaged: Bool           // dispatch the QA13 EMA blend
        let resetState: Bool        // disengage: ema := live ROI (replay.py:591-594)
        let takeLockSnapshot: Bool  // QA15 byte-freeze copy (after the EMA write)
        let blendLock: Bool         // QA15 frozen-core blend (after the snapshot)
    }

    // GPU twins of `emaState != nil` / `lockRun` / `lockState != nil`. A
    // Serve320SilenceEMA instance must be driven by exactly one path — apply()
    // (CPU) or stepMetal() (GPU) — for its whole life; the product pipelines
    // pick the path once at prepare time.
    private var metalHasState = false
    private var metalLockRun = 0
    private var metalLockHeld = false

    /// Decision-only mirror of `apply` for the Metal path: identical detector,
    /// identical engage/disengage/hard-lock state machine and audit rows; the
    /// ROI pixel work is dispatched by the caller in the returned order
    /// (reg capture -> EMA -> snapshot -> lock blend, or reg capture -> reset).
    func stepMetal(frame i: Int, aperture ap: Float) -> MetalStep {
        let silent = detectSilence(frame: i, aperture: ap)
        let step: MetalStep
        if silent, metalHasState {
            var snapshot = false, blend = false
            if hardLockEnabled {
                // replay.py:58-66 _hard_lock_step(True) + 585-594
                metalLockRun += 1
                if metalLockRun >= Self.hardLockHold {
                    if !metalLockHeld {
                        metalLockHeld = true
                        snapshot = true
                    }
                    blend = true
                }
            }
            step = MetalStep(engaged: true, resetState: false,
                             takeLockSnapshot: snapshot, blendLock: blend)
        } else {
            // replay.py:591-594 — track the live frame; INSTANT lock release
            metalHasState = true
            if hardLockEnabled {
                metalLockRun = 0
                metalLockHeld = false
            }
            step = MetalStep(engaged: false, resetState: true,
                             takeLockSnapshot: false, blendLock: false)
        }
        audit.append(AuditFrame(aperture: ap, energy: energy[i], silent: silent,
                                engaged: step.engaged, locked: metalLockHeld))
        return step
    }
}
