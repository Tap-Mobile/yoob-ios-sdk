//
//  Serve320Compositor.swift
//  The QA9 serve compositor for the 320 lane: aperture-gated contour-intersected
//  support -> chin cap (QA7) -> tonematch (QA8) -> skinmask (QA9) ->
//  canonical_composite_native onto the 1080x1920 idle canvas.
//  Mirrors evaluation/per_frame_support.py, evaluation/canonical_composite.py,
//  and the render_utils.py helpers — cited per function. Params come from
//  bundle/meta.json serve_config.qa9_compositor.
//

import Foundation
import simd

struct Serve320MouthInkMetrics {
    let boundsCenterX: Float
    let centroidX: Float
    let minX: Int
    let maxX: Int
    let minY: Int
    let maxY: Int
    let pixelCount: Int
    let componentCount: Int

    var width: Int { maxX - minX + 1 }
    var height: Int { maxY - minY + 1 }
}

/// Order-free bounding-box reduction for row-chunked scans. Each fanned-out
/// chunk merges its local box under the lock; min/max make the merge order
/// irrelevant, so the result is identical to the serial scan.
final class Serve320ParallelBoxes {
    private let lock = NSLock()
    private var x0: Int
    private var x1 = -1
    private var y0: Int
    private var y1 = -1

    init(rows: Int) {
        x0 = rows
        y0 = rows
    }

    func merge(x0 mx0: Int, x1 mx1: Int, y0 my0: Int, y1 my1: Int) {
        guard mx1 >= mx0, my1 >= my0 else { return }
        lock.lock()
        x0 = min(x0, mx0); x1 = max(x1, mx1)
        y0 = min(y0, my0); y1 = max(y1, my1)
        lock.unlock()
    }

    func box(defaultX: Int, defaultY: Int) -> (Int, Int, Int, Int) {
        lock.lock()
        defer { lock.unlock() }
        return x1 >= 0 ? (x0, x1, y0, y1) : (defaultX, -1, defaultY, -1)
    }
}

enum Serve320Compositor {

    static let res = 320

    static func apertureSupportDepth(_ aperture: Float, maxBandPx: Int = 70,
                                     closedAp: Float = 0.20,
                                     openAp: Float = 0.80) -> Int {
        let frac = min(max((aperture - closedAp) / max(openAp - closedAp, 1e-6), 0), 1)
        return Int((frac * Float(maxBandPx)).rounded(.toNearestOrEven))
    }

    /// Returns a translation request only when the mouth bounds and the
    /// pigment mass independently agree on its direction. Bounds alone can be
    /// fooled by a wide asymmetric lip shape; translating on that signal was
    /// the source of the occasional left/right jump and shifted lower-face
    /// patch. The smaller agreeing residual is deliberately conservative.
    static func guardedMouthInkShift(boundsDelta: Float,
                                      centroidDelta: Float,
                                      deadband: Float = 0.5) -> Float {
        guard boundsDelta.isFinite, centroidDelta.isFinite,
              abs(boundsDelta) >= deadband,
              abs(centroidDelta) >= deadband,
              boundsDelta.sign == centroidDelta.sign else { return 0 }
        return boundsDelta.sign == .minus
            ? -min(abs(boundsDelta), abs(centroidDelta))
            : min(abs(boundsDelta), abs(centroidDelta))
    }

    static func boundedMouthInkShift(requested: Float, limit: Int) -> Int {
        let safeLimit = max(limit, 0)
        return min(max(
            Int(requested.rounded(.toNearestOrEven)),
            -safeLimit), safeLimit)
    }

    // MARK: evaluation/per_frame_support.py:62-90 (aperture_gated_support)
    /// support ∩ aperture-dilated face contour, feathered (2px), hole-hard.
    /// contourU8: the frame's PRE-WARPED 320 contour (>= 128 == inside).
    static func apertureGatedSupport(support: [Float],
                                     contourU8: UnsafeRawBufferPointer,
                                     aperture: Float,
                                     hole: [Float],
                                     maxBandPx: Int = 70,
                                     closedAp: Float = 0.20,
                                     openAp: Float = 0.80,
                                     featherPx: Int = 2) -> [Float] {
        let S = res
        // per_frame_support.py:81-82 — np.round is half-even.
        let d = apertureSupportDepth(aperture, maxBandPx: maxBandPx,
                                     closedAp: closedAp, openAp: openAp)
        // :83 — inside = fc >= 0.5  (u8/255 >= 0.5  <=>  u8 >= 128)
        // Full-plane scan: row chunks fan out, each reduces a local bbox and
        // the boxes merge afterwards (min/max are order-free).
        var inside = [Float](repeating: 0, count: S * S)
        var bx0 = S, bx1 = -1, by0 = S, by1 = -1
        let chunkBoxes = Serve320ParallelBoxes(rows: S)
        inside.withUnsafeMutableBufferPointer { buf in
            let ip = buf.baseAddress!
            Serve320ImageOps.fanOutLines(0, S - 1) { cy0, cy1 in
                var lx0 = S, lx1 = -1, ly0 = S, ly1 = -1
                for y in cy0...cy1 {
                    let row = y * S
                    for x in 0..<S {
                        let value: Float = contourU8[row + x] >= 128 ? 1 : 0
                        ip[row + x] = value
                        if value > 0 {
                            lx0 = min(lx0, x); lx1 = max(lx1, x)
                            ly0 = min(ly0, y); ly1 = max(ly1, y)
                        }
                    }
                }
                chunkBoxes.merge(x0: lx0, x1: lx1, y0: ly0, y1: ly1)
            }
        }
        (bx0, bx1, by0, by1) = chunkBoxes.box(defaultX: S, defaultY: S)
        var out = [Float](repeating: 0, count: S * S)
        guard bx1 >= bx0, by1 >= by0 else { return out }
        // :84-88 — downward-only dilation by d px
        let dilationWindow = Serve320ImageOps.RectWindow(
            x0: bx0, y0: by0, x1: bx1, y1: min(S - 1, by1 + d))
        let ext = Serve320ImageOps.dilateDownBinary(
            inside, w: S, h: S, d: d, window: dilationWindow)

        var sx0 = S, sx1 = -1, sy0 = S, sy1 = -1
        let extBoxes = Serve320ParallelBoxes(rows: S)
        ext.withUnsafeBufferPointer { buf in
            let ep = buf.baseAddress!
            Serve320ImageOps.fanOutLines(dilationWindow.y0, dilationWindow.y1) { cy0, cy1 in
                var lx0 = S, lx1 = -1, ly0 = S, ly1 = -1
                for y in cy0...cy1 {
                    let row = y * S
                    for x in dilationWindow.x0...dilationWindow.x1 where ep[row + x] >= 0.5 {
                        lx0 = min(lx0, x); lx1 = max(lx1, x)
                        ly0 = min(ly0, y); ly1 = max(ly1, y)
                    }
                }
                extBoxes.merge(x0: lx0, x1: lx1, y0: ly0, y1: ly1)
            }
        }
        (sx0, sx1, sy0, sy1) = extBoxes.box(defaultX: S, defaultY: S)
        guard sx1 >= sx0, sy1 >= sy0 else { return out }
        let supportWindow = Serve320ImageOps.RectWindow(
            x0: sx0, y0: sy0, x1: sx1, y1: sy1)

        if featherPx > 0 {
            let sigma = max(Float(featherPx) * 0.6, 0.5)
            let blurRadius = (Int((sigma * 4 * 2 + 1).rounded()) | 1) / 2
            let erosionWindow = Serve320ImageOps.RectWindow(
                x0: max(0, min(sx0 - blurRadius, 2 * (S - 1) - (sx1 + blurRadius))),
                y0: max(0, min(sy0 - blurRadius, 2 * (S - 1) - (sy1 + blurRadius))),
                x1: min(S - 1, max(sx1 + blurRadius, blurRadius - sx0)),
                y1: min(S - 1, max(sy1 + blurRadius, blurRadius - sy0)))
            let eroded = Serve320ImageOps.erodeEllipse5x5(
                ext, w: S, h: S, window: erosionWindow)
            let blurred = Serve320ImageOps.gaussianBlur(
                eroded, w: S, h: S, sigma: sigma, window: supportWindow)
            out.withUnsafeMutableBufferPointer { buf in
                let op = buf.baseAddress!
                Serve320ImageOps.fanOutLines(sy0, sy1) { cy0, cy1 in
                    for y in cy0...cy1 {
                        let row = y * S
                        for x in sx0...sx1 {
                            let i = row + x
                            var ramp = min(blurred[i], ext[i])
                            if hole[i] >= 0.5 { ramp = ext[i] }
                            op[i] = ext[i] < 0.5 ? 0 : min(support[i], ramp)
                        }
                    }
                }
            }
        } else {
            out.withUnsafeMutableBufferPointer { buf in
                let op = buf.baseAddress!
                Serve320ImageOps.fanOutLines(sy0, sy1) { cy0, cy1 in
                    for y in cy0...cy1 {
                        let row = y * S
                        for x in sx0...sx1 {
                            let i = row + x
                            op[i] = min(support[i], ext[i])
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: evaluation/per_frame_support.py:23-59 (contour_intersected_support)
    static func contourIntersectedSupport(support: [Float], contour: [Float],
                                          featherPx: Int, hole: [Float]?) -> [Float] {
        let S = res
        // :45 — inside = fc >= 0.5
        let inside = contour.map { $0 >= 0.5 ? Float(1) : Float(0) }
        var ramp: [Float]
        if featherPx > 0 {
            // :47-49 — erode(5x5 ellipse) -> GaussianBlur sigma max(feather*0.6, 0.5)
            let eroded = Serve320ImageOps.erodeEllipse5x5(inside, w: S, h: S)
            let sigma = max(Float(featherPx) * 0.6, 0.5)
            ramp = Serve320ImageOps.gaussianBlur(eroded, w: S, h: S, sigma: sigma)
            for i in 0..<(S * S) { ramp[i] = min(ramp[i], inside[i]) }   // inward feather
        } else {
            ramp = inside
        }
        if let hole {
            // :52-56 — NO partial alpha inside the hole (hard 0/1 there)
            for i in 0..<(S * S) {
                if hole[i] >= 0.5 { ramp[i] = inside[i] }
            }
        }
        var out = [Float](repeating: 0, count: S * S)
        for i in 0..<(S * S) {
            out[i] = min(support[i], ramp[i])           // :57
            if inside[i] < 0.5 { out[i] = 0 }           // :58 bit-exact outside the contour
        }
        return out
    }

    // MARK: render_utils.lip_cap_weight (arm_a/render_utils.py:353-367)
    /// QA7: support row weight — 1 at/above capBase+margin, ramp to 0 over feather rows.
    static func lipCapWeight(capBase: Float, marginPx: Float = 4.0,
                             featherPx: Float = 14.0) -> [Float] {
        let S = res
        let y0 = capBase + marginPx
        var w = [Float](repeating: 0, count: S)
        for y in 0..<S {
            w[y] = min(max(1 - (Float(y) - y0) / max(featherPx, 1), 0), 1)
        }
        return w
    }

    // MARK: render_utils.cap_dc_correct (arm_a/render_utils.py:370-401)
    /// QA8: DC-match the renderer skin to the host chin at the cap seam.
    /// pred/host: BGR u8 HWC (320,320,3). mask = uncapped aperture-gated support.
    /// np round/int semantics: cap/lip via round-half-even; cx truncated.
    static func capDcCorrect(pred: inout [UInt8], host: [UInt8],
                             lipY: Float, capY: Float,
                             mask: [Float], cx: Float, k: Int = 14) {
        let S = res
        let cap = Int(capY.rounded(.toNearestOrEven))
        let lip = Int(lipY.rounded(.toNearestOrEven))
        let half = Int(0.22 * Float(S))                       // :386 (int() truncates)
        let y0 = cap, y1 = min(S, cap + k)
        let x0 = max(0, Int(cx) - half), x1 = min(S, Int(cx) + half)
        guard y1 > y0, x1 > x0 else { return }                // :388-389
        var hb: [SIMD3<Float>] = []
        var pb: [SIMD3<Float>] = []
        hb.reserveCapacity((y1 - y0) * (x1 - x0))
        pb.reserveCapacity(hb.capacity)
        for y in y0..<y1 {
            for x in x0..<x1 {
                let m = mask[y * S + x] > 0.1                 // :392
                let base = (y * S + x) * 3
                let hpx = SIMD3<Float>(Float(host[base]), Float(host[base + 1]), Float(host[base + 2]))
                guard m, max(hpx.x, max(hpx.y, hpx.z)) > 120 else { continue }  // :393-394 bright skin only
                hb.append(hpx)
                pb.append(SIMD3<Float>(Float(pred[base]), Float(pred[base + 1]), Float(pred[base + 2])))
            }
        }
        guard hb.count >= 30 else { return }                  // :395-396
        var delta = SIMD3<Float>(
            Serve320ImageOps.median(hb.map { $0.x }) - Serve320ImageOps.median(pb.map { $0.x }),
            Serve320ImageOps.median(hb.map { $0.y }) - Serve320ImageOps.median(pb.map { $0.y }),
            Serve320ImageOps.median(hb.map { $0.z }) - Serve320ImageOps.median(pb.map { $0.z }))
        delta = simd_min(simd_max(delta, SIMD3<Float>(-35, -35, -35)),
                         SIMD3<Float>(35, 35, 35))            // :397
        // :398-401 — taper 0 at the lip -> 1 at the cap; +delta; clip; astype(u8) truncates
        let denom = Float(max(cap - lip, 1))
        for y in 0..<S {
            let vt = min(max((Float(y) - Float(lip)) / denom, 0), 1)
            guard vt > 0 else { continue }
            let d0 = delta.x * vt, d1 = delta.y * vt, d2 = delta.z * vt
            let row = y * S
            for x in 0..<S {
                let base = (row + x) * 3
                pred[base] = UInt8(min(max(pred[base].floatPlus(d0), 0), 255))
                pred[base + 1] = UInt8(min(max(pred[base + 1].floatPlus(d1), 0), 255))
                pred[base + 2] = UInt8(min(max(pred[base + 2].floatPlus(d2), 0), 255))
            }
        }
    }

    // MARK: render_utils.host_skin_mask (arm_a/render_utils.py:404-424)
    /// QA9: 1 where the host crop is chin-skin-colored (mouth polygon exempted
    /// by the caller). k=14 rows below capY, central column +-0.22*S, tol 60,
    /// morphological close 5x5.
    static func hostSkinMask(host: [UInt8], cx: Float, capY: Float,
                             k: Int = 14, tol: Float = 60, closePx: Int = 5) -> [Float] {
        let S = res
        let cap = Int(capY.rounded(.toNearestOrEven))
        let half = Int(0.22 * Float(S))
        let y0 = cap, y1 = min(S, cap + k)
        let x0 = max(0, Int(cx) - half), x1 = min(S, Int(cx) + half)
        var patch: [SIMD3<Float>] = []
        if y1 > y0, x1 > x0 {
            patch.reserveCapacity((y1 - y0) * (x1 - x0))
            for y in y0..<y1 {
                for x in x0..<x1 {
                    let base = (y * S + x) * 3
                    patch.append(SIMD3<Float>(Float(host[base]), Float(host[base + 1]),
                                              Float(host[base + 2])))
                }
            }
        }
        // :418-419 — reference tone = median of the central BRIGHT patch (>30 px),
        // else median of the whole crop. Medians via selection (bit-exact order
        // statistics); the dark-host fallback used to run THREE full sorts over
        // the 102k-pixel crop.
        let bright = patch.filter { max($0.x, max($0.y, $0.z)) > 120 }
        let ref: SIMD3<Float>
        if bright.count > 30 {
            var b = bright.map { $0.x }
            var g = bright.map { $0.y }
            var r = bright.map { $0.z }
            ref = SIMD3<Float>(Serve320ImageOps.medianInPlace(&b),
                               Serve320ImageOps.medianInPlace(&g),
                               Serve320ImageOps.medianInPlace(&r))
        } else {
            let count = S * S
            var b = [Float](repeating: 0, count: count)
            var g = [Float](repeating: 0, count: count)
            var r = [Float](repeating: 0, count: count)
            for px in 0..<count {
                b[px] = Float(host[px * 3])
                g[px] = Float(host[px * 3 + 1])
                r[px] = Float(host[px * 3 + 2])
            }
            ref = SIMD3<Float>(Serve320ImageOps.medianInPlace(&b),
                               Serve320ImageOps.medianInPlace(&g),
                               Serve320ImageOps.medianInPlace(&r))
        }
        // :420-421 — dist < tol. Full-plane scan fans out with a merged bbox.
        var m = [Float](repeating: 0, count: S * S)
        var sx0 = S, sx1 = -1, sy0 = S, sy1 = -1
        let tolSquared = tol * tol
        let skinBoxes = Serve320ParallelBoxes(rows: S)
        host.withUnsafeBufferPointer { hostBuf in
            m.withUnsafeMutableBufferPointer { maskBuf in
                let hp = hostBuf.baseAddress!
                let mp = maskBuf.baseAddress!
                Serve320ImageOps.fanOutLines(0, S - 1) { cy0, cy1 in
                    var lx0 = S, lx1 = -1, ly0 = S, ly1 = -1
                    for y in cy0...cy1 {
                        let row = y * S
                        for x in 0..<S {
                            let base = (row + x) * 3
                            let db = Float(hp[base]) - ref.x
                            let dg = Float(hp[base + 1]) - ref.y
                            let dr = Float(hp[base + 2]) - ref.z
                            if db * db + dg * dg + dr * dr < tolSquared {
                                mp[row + x] = 1
                                lx0 = min(lx0, x); lx1 = max(lx1, x)
                                ly0 = min(ly0, y); ly1 = max(ly1, y)
                            }
                        }
                    }
                    skinBoxes.merge(x0: lx0, x1: lx1, y0: ly0, y1: ly1)
                }
            }
        }
        (sx0, sx1, sy0, sy1) = skinBoxes.box(defaultX: S, defaultY: S)
        if closePx > 0 {
            guard sx1 >= sx0, sy1 >= sy0 else { return m }
            let radius = closePx / 2
            let window = Serve320ImageOps.RectWindow(
                x0: max(0, sx0 - radius), y0: max(0, sy0 - radius),
                x1: min(S - 1, sx1 + radius), y1: min(S - 1, sy1 + radius))
            m = Serve320ImageOps.morphCloseRect(
                m, w: S, h: S, k: closePx, window: window)   // :422-423
        }
        return m
    }

    private static func wholeCropPixels(_ host: [UInt8]) -> [SIMD3<Float>] {
        var out: [SIMD3<Float>] = []
        out.reserveCapacity(res * res)
        for i in 0..<(res * res) {
            out.append(SIMD3<Float>(Float(host[i * 3]), Float(host[i * 3 + 1]),
                                    Float(host[i * 3 + 2])))
        }
        return out
    }

    /// Protect the output from carrying two mouths at once. QA9 normally refuses
    /// to paint renderer pixels over non-skin host art, with the target landmark
    /// hull as the sole exception. When the target and idle mouths do not overlap,
    /// that leaves pieces of the idle lip/teeth untouched. Restrict the second
    /// exception to non-skin pixels in the tracked host-mouth box and inside the
    /// renderer input hole; hair/collar protection remains unchanged.
    static func hostMouthResidualMask(skin: [Float], hole: [Float],
                                      contourU8: UnsafeRawBufferPointer,
                                      center: (x: Float, y: Float),
                                      lowerLipY: Float, anchorWidth: Float) -> [Float] {
        let S = res
        let radiusX = 0.56 * anchorWidth
        let top = center.y - 0.16 * anchorWidth
        let bottom = lowerLipY + 0.035 * anchorWidth
        let ellipseY = (top + bottom) * 0.5
        let radiusY = max((bottom - top) * 0.5, 1)
        let x0 = max(0, Int(floor(center.x - radiusX)))
        let x1 = min(S, Int(ceil(center.x + radiusX)) + 1)
        let y0 = max(0, Int(floor(top)))
        let y1 = min(S, Int(ceil(bottom)) + 1)
        let faceMargin = 8
        let faceSamples = [
            (-faceMargin, -faceMargin), (0, -faceMargin), (faceMargin, -faceMargin),
            (-faceMargin, 0), (0, 0), (faceMargin, 0),
            (-faceMargin, faceMargin), (0, faceMargin), (faceMargin, faceMargin),
        ]
        var out = [Float](repeating: 0, count: S * S)
        guard x1 > x0, y1 > y0 else { return out }
        for y in y0..<y1 {
            let row = y * S
            for x in x0..<x1 {
                let px = row + x
                let nx = (Float(x) - center.x) / radiusX
                let ny = (Float(y) - ellipseY) / radiusY
                guard nx * nx + ny * ny <= 1,
                      hole[px] >= 0.5, skin[px] < 0.5 else { continue }
                var safelyInsideFace = true
                for (dx, dy) in faceSamples {
                    let xx = x + dx, yy = y + dy
                    if xx < 0 || xx >= S || yy < 0 || yy >= S ||
                        contourU8[yy * S + xx] < 128 {
                        safelyInsideFace = false
                        break
                    }
                }
                if safelyInsideFace { out[px] = 1 }
            }
        }
        return out
    }

    /// Avatar-specific visible mouth-pigment center used by the final-pixel
    /// registration gate. The thresholds intentionally match the independent
    /// Mac-side diagnostic: red lip pigment or a dark mouth cavity, followed by
    /// eight-connected component rejection around the tracked host mouth.
    static func mouthInkCenterX(imageBGR: [UInt8], width: Int, height: Int,
                                expectedX: Float, expectedY: Float) -> Float? {
        mouthInkMetrics(imageBGR: imageBGR, width: width, height: height,
                        expectedX: expectedX, expectedY: expectedY)?.boundsCenterX
    }

    static func mouthInkCenterX(imageBGRA: [UInt8], width: Int, height: Int,
                                expectedX: Float, expectedY: Float) -> Float? {
        mouthInkMetrics(imageBGRA: imageBGRA, width: width, height: height,
                        expectedX: expectedX, expectedY: expectedY)?.boundsCenterX
    }

    static func mouthInkMetrics(imageBGR: [UInt8], width: Int, height: Int,
                                expectedX: Float, expectedY: Float) -> Serve320MouthInkMetrics? {
        mouthInkMetrics(bytes: imageBGR, width: width, height: height,
                        pixelStride: 3, redOffset: 2, greenOffset: 1, blueOffset: 0,
                        expectedX: expectedX, expectedY: expectedY)
    }

    static func mouthInkMetrics(imageBGRA: [UInt8], width: Int, height: Int,
                                expectedX: Float, expectedY: Float) -> Serve320MouthInkMetrics? {
        mouthInkMetrics(bytes: imageBGRA, width: width, height: height,
                        pixelStride: 4, redOffset: 2, greenOffset: 1, blueOffset: 0,
                        expectedX: expectedX, expectedY: expectedY)
    }

    /// Positive dx moves visible renderer pixels right. Support remains fixed
    /// in host space so jaw/old-mouth ownership is not weakened.
    static func translateBGRHorizontally(_ src: [UInt8], width: Int, height: Int,
                                         dx: Int) -> [UInt8] {
        guard dx != 0 else { return src }
        var dst = [UInt8](repeating: 0, count: src.count)
        for y in 0..<height {
            let row = y * width
            for x in 0..<width {
                let sourceX = min(max(x - dx, 0), width - 1)
                let sourceBase = (row + sourceX) * 3
                let destinationBase = (row + x) * 3
                dst[destinationBase] = src[sourceBase]
                dst[destinationBase + 1] = src[sourceBase + 1]
                dst[destinationBase + 2] = src[sourceBase + 2]
            }
        }
        return dst
    }

    private static func mouthInkMetrics(bytes: [UInt8], width: Int, height: Int,
                                       pixelStride: Int,
                                       redOffset: Int, greenOffset: Int, blueOffset: Int,
                                       expectedX: Float,
                                       expectedY: Float) -> Serve320MouthInkMetrics? {
        guard width > 0, height > 0,
              bytes.count >= width * height * pixelStride else { return nil }
        let x0 = max(0, Int(ceil(expectedX - 70)))
        let x1 = min(width - 1, Int(floor(expectedX + 70)))
        let y0 = max(0, Int(ceil(expectedY - 30)))
        let y1 = min(height - 1, Int(floor(expectedY + 48)))
        guard x1 >= x0, y1 >= y0 else { return nil }

        let localWidth = x1 - x0 + 1
        let localHeight = y1 - y0 + 1
        var ink = [UInt8](repeating: 0, count: localWidth * localHeight)
        for localY in 0..<localHeight {
            let y = y0 + localY
            for localX in 0..<localWidth {
                let x = x0 + localX
                let base = (y * width + x) * pixelStride
                let r = Int(bytes[base + redOffset])
                let g = Int(bytes[base + greenOffset])
                let b = Int(bytes[base + blueOffset])
                let isPigment = (r - g > 14 && r - b > 8 && g < 215)
                    || (r < 145 && g < 105 && b < 120)
                if isPigment { ink[localY * localWidth + localX] = 1 }
            }
        }

        var seen = [UInt8](repeating: 0, count: ink.count)
        var queue: [Int] = []
        var histogram = [Int](repeating: 0, count: width)
        var keptCount = 0
        var keptComponents = 0
        var keptMinX = Int.max, keptMaxX = Int.min
        var keptMinY = Int.max, keptMaxY = Int.min
        var keptXSum: Int64 = 0
        for seed in ink.indices where ink[seed] != 0 && seen[seed] == 0 {
            queue.removeAll(keepingCapacity: true)
            queue.append(seed)
            seen[seed] = 1
            var head = 0
            var minX = Int.max, maxX = Int.min
            var minY = Int.max, maxY = Int.min
            while head < queue.count {
                let index = queue[head]
                head += 1
                let localY = index / localWidth
                let localX = index - localY * localWidth
                let x = x0 + localX
                let y = y0 + localY
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let nextX = localX + dx, nextY = localY + dy
                        guard nextX >= 0, nextX < localWidth,
                              nextY >= 0, nextY < localHeight else { continue }
                        let next = nextY * localWidth + nextX
                        if ink[next] != 0, seen[next] == 0 {
                            seen[next] = 1
                            queue.append(next)
                        }
                    }
                }
            }

            let componentCenterX = Float(minX + maxX) * 0.5
            let componentCenterY = Float(minY + maxY) * 0.5
            guard queue.count >= 8,
                  maxX - minX >= 5,
                  abs(componentCenterX - expectedX) <= 45,
                  abs(componentCenterY - expectedY) <= 30 else { continue }
            keptComponents += 1
            keptMinX = min(keptMinX, minX); keptMaxX = max(keptMaxX, maxX)
            keptMinY = min(keptMinY, minY); keptMaxY = max(keptMaxY, maxY)
            for index in queue {
                let localX = index % localWidth
                let x = x0 + localX
                histogram[x] += 1
                keptXSum += Int64(x)
                keptCount += 1
            }
        }
        guard keptCount > 0,
              let p01 = histogramPercentile(histogram, count: keptCount, q: 0.01),
              let p99 = histogramPercentile(histogram, count: keptCount, q: 0.99)
        else { return nil }
        return Serve320MouthInkMetrics(
            boundsCenterX: (p01 + p99) * 0.5,
            centroidX: Float(keptXSum) / Float(keptCount),
            minX: keptMinX,
            maxX: keptMaxX,
            minY: keptMinY,
            maxY: keptMaxY,
            pixelCount: keptCount,
            componentCount: keptComponents
        )
    }

    /// Mean absolute 4-neighbour Laplacian over the detected mouth bounds.
    /// It is intentionally a diagnostic, not a beautification filter: low-tail
    /// open-mouth frames are dumped for review instead of being sharpened blind.
    static func mouthSharpness(imageBGR: [UInt8], width: Int, height: Int,
                               metrics: Serve320MouthInkMetrics) -> Float? {
        let x0 = max(1, metrics.minX), x1 = min(width - 2, metrics.maxX)
        let y0 = max(1, metrics.minY), y1 = min(height - 2, metrics.maxY)
        guard x1 >= x0, y1 >= y0 else { return nil }
        var total: Float = 0
        var count = 0
        func luma(_ x: Int, _ y: Int) -> Float {
            let base = (y * width + x) * 3
            return 0.114 * Float(imageBGR[base])
                + 0.587 * Float(imageBGR[base + 1])
                + 0.299 * Float(imageBGR[base + 2])
        }
        for y in y0...y1 {
            for x in x0...x1 {
                let center = luma(x, y)
                let laplacian = 4 * center - luma(x - 1, y) - luma(x + 1, y)
                    - luma(x, y - 1) - luma(x, y + 1)
                total += abs(laplacian)
                count += 1
            }
        }
        return count > 0 ? total / Float(count) : nil
    }

    private static func histogramPercentile(_ histogram: [Int], count: Int,
                                            q: Float) -> Float? {
        guard count > 0 else { return nil }
        let position = q * Float(count - 1)
        let lowRank = Int(floor(position))
        let highRank = Int(ceil(position))
        func value(at rank: Int) -> Float? {
            var seen = 0
            for (value, frequency) in histogram.enumerated() where frequency > 0 {
                if rank < seen + frequency { return Float(value) }
                seen += frequency
            }
            return nil
        }
        guard let low = value(at: lowRank), let high = value(at: highRank) else { return nil }
        return low + (high - low) * (position - Float(lowRank))
    }

    // MARK: evaluation/canonical_composite.py:54-81 (canonical_composite_native)
    /// Upsample pred + support to the stab box res FIRST, then blend once:
    /// out = region*(1-sup) + pred*sup (np.rint = round-half-even, clip, u8).
    /// canvas: BGRA bytes (1080x1920), mutated in the box region only.
    /// pred: RAW renderer output BGR u8 HWC 320x320x3 (no pre-blend).
    static func canonicalCompositeNative(canvas: inout [UInt8], canvasW: Int, canvasH: Int,
                                         predBGR: [UInt8], support: [Float],
                                         box: (x0: Int, y0: Int, x1: Int, y1: Int)) {
        let S = res
        let bw = box.x1 - box.x0, bh = box.y1 - box.y0
        guard bw > 0, bh > 0, box.x0 >= 0, box.y0 >= 0,
              box.x1 <= canvasW, box.y1 <= canvasH else { return }
        // :76 — pred_box = cv2.resize(pred, (bw,bh), INTER_LINEAR) per channel (HWC BGR).
        // The three channels are independent, so deinterleave/resize/reinterleave
        // runs concurrently. The previous version also allocated a fresh plane and
        // a fresh result array per channel — six heap allocations and ~600 KB of
        // churn on every frame.
        var predBox = [UInt8](repeating: 0, count: bw * bh * 3)
        var planes = [[UInt8]](repeating: [], count: 3)
        planes.withUnsafeMutableBufferPointer { slot in
            DispatchQueue.concurrentPerform(iterations: 3) { c in
                var plane = [UInt8](repeating: 0, count: S * S)
                predBGR.withUnsafeBufferPointer { source in
                    plane.withUnsafeMutableBufferPointer { out in
                        for i in 0..<(S * S) { out[i] = source[i * 3 + c] }
                    }
                }
                slot[c] = Serve320ImageOps.resizeBilinearU8(plane, sw: S, sh: S, dw: bw, dh: bh)
            }
        }
        predBox.withUnsafeMutableBufferPointer { out in
            for c in 0..<3 {
                planes[c].withUnsafeBufferPointer { up in
                    for i in 0..<(bw * bh) { out[i * 3 + c] = up[i] }
                }
            }
        }
        // :77 — sup_box = cv2.resize(sup, (bw,bh), INTER_LINEAR)
        let supBox = Serve320ImageOps.resizeBilinearFloat(support, sw: S, sh: S, dw: bw, dh: bh)
        // :78-80 — blend once over the box region (canvas is BGRA; B,G,R at +0,+1,+2).
        // Each row touches a disjoint canvas range, so rows fan out across cores.
        // Per-pixel arithmetic and rounding are unchanged, so the composite stays
        // bit-identical to the b200 reference.
        canvas.withUnsafeMutableBufferPointer { canvasBuffer in
            let pixels = canvasBuffer.baseAddress!
            predBox.withUnsafeBufferPointer { pred in
                supBox.withUnsafeBufferPointer { sup in
                    let body = { (y: Int) in
                        let canvasRow = (box.y0 + y) * canvasW + box.x0
                        let supRow = y * bw
                        for x in 0..<bw {
                            let weight = sup[supRow + x]
                            guard weight > 0 else { continue }
                            let cb = canvasRow * 4 + x * 4
                            let pb = (supRow + x) * 3
                            for c in 0..<3 {
                                let region = Float(pixels[cb + c])
                                let p = Float(pred[pb + c])
                                let v = (region * (1 - weight) + p * weight)
                                    .rounded(.toNearestOrEven)
                                pixels[cb + c] = UInt8(min(max(v, 0), 255))
                            }
                        }
                    }
                    if bh >= Serve320ImageOps.parallelRowThreshold {
                        DispatchQueue.concurrentPerform(iterations: bh, execute: body)
                    } else {
                        for y in 0..<bh { body(y) }
                    }
                }
            }
        }
    }
}

private extension UInt8 {
    /// float + delta, for cap_dc_correct's astype(uint8) truncation semantics
    /// (render_utils.py:400-401: np.clip(f32,0,255).astype(np.uint8) — C truncation).
    func floatPlus(_ d: Float) -> Float { Float(self) + d }
}
