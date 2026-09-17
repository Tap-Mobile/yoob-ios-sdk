//
//  Serve320Math.swift
//  Geometry / landmark math for the Serve320 lane. Every function mirrors a
//  specific python function in talking-head-v3-ofir-v2 (cited per function);
//  keep the math in lockstep with those references.
//

import Foundation
import Accelerate
import simd

enum Serve320Math {

    // MARK: contracts.reconstruct_geometry (arm_a/contracts.py:150-154)
    /// scores6 (6) standardized -> canonical 40D geometry (fp64 accumulation, fp32 out).
    static func reconstructGeometry(_ scores6: [Float], pca: Serve320Pca) -> [Float] {
        var out = [Float](repeating: 0, count: 40)
        var acc = [Double](repeating: 0, count: 40)
        for k in 0..<6 {
            let s = Double(scores6[k]) * pca.score_std[k]
            guard s != 0 else { continue }
            let row = pca.components[k]
            for j in 0..<40 {
                acc[j] += s * row[j]
            }
        }
        for j in 0..<40 {
            out[j] = Float(pca.geometry_mean[j] + acc[j])
        }
        return out
    }

    // MARK: render_utils.decode_points_perframe (arm_a/render_utils.py:259-272)
    /// pred6 (6) + anchor [mid_x, mid_y, width, angle] -> 20x2 crop-space landmarks.
    static func decodePointsPerframe(_ scores6: [Float],
                                     anchor: (midX: Float, midY: Float, width: Float, angle: Float),
                                     pca: Serve320Pca) -> [SIMD2<Float>] {
        let recon = reconstructGeometry(scores6, pca: pca)   // canonical unit-width, (20,2)
        let ca = cos(anchor.angle), sa = sin(anchor.angle)
        var pts = [SIMD2<Float>](repeating: .zero, count: 20)
        for i in 0..<20 {
            let x = recon[2 * i], y = recon[2 * i + 1]
            // R(angle) @ p (render_utils.py:268-271)
            let rx = ca * x - sa * y
            let ry = sa * x + ca * y
            pts[i] = SIMD2(anchor.midX + anchor.width * rx,
                           anchor.midY + anchor.width * ry)
        }
        return pts
    }

    // MARK: contracts.aperture_from_geometry (arm_a/contracts.py:70-85)
    /// geom40 = canonical 40D (from reconstructGeometry). corners (0,6),
    /// inner vertical pairs (13,19),(14,18),(15,17), full-open ratio 0.35.
    static func apertureFromGeometry(_ geom40: [Float]) -> Float {
        func px(_ i: Int) -> Float { geom40[2 * i] }
        func py(_ i: Int) -> Float { geom40[2 * i + 1] }
        let width = hypot(px(6) - px(0), py(6) - py(0))
        guard width > 1e-6 else { return 0 }
        let gaps: [Float] = [py(19) - py(13), py(18) - py(14), py(17) - py(15)]
            .map { max(0, $0) }
        let median = gaps.sorted()[1]
        return min(1, max(0, (median / width) / 0.35))
    }

    /// Convenience: aperture of a pred6 row (contracts pipeline via reconstruct).
    static func aperture(_ scores6: [Float], pca: Serve320Pca) -> Float {
        apertureFromGeometry(reconstructGeometry(scores6, pca: pca))
    }

    // MARK: replay.r2b_blend (arm_a/replay.py:47-55)
    /// Blend landmark VERTICALS toward the frozen closed template (centroid-aligned
    /// y only). gate = thresholded sigmoid prob (0 below thr), strength = 0.9.
    static func r2bBlend(_ pts: [SIMD2<Float>], gate: Float,
                         tmplYoff: [Float], strength: Float) -> [SIMD2<Float>] {
        let g = min(1, max(0, gate)) * strength
        guard g > 0 else { return pts }
        var cenY: Float = 0
        for p in pts { cenY += p.y }
        cenY /= Float(pts.count)
        var out = pts
        for i in 0..<pts.count {
            out[i].y = cenY + (1 - g) * (pts[i].y - cenY) + g * tmplYoff[i]
        }
        return out
    }

    // MARK: replay._r2b_gate_for thresholding (arm_a/replay.py:388-401)
    static func r2bSigmoid(_ logit: Float) -> Float {
        1 / (1 + exp(-logit))
    }

    /// sigmoid(logit) > thr ? sigmoid : 0 (per-frame scalar here).
    static func r2bGate(logit: Float, threshold: Float) -> Float {
        let g = r2bSigmoid(logit)
        return g > threshold ? g : 0
    }

    // MARK: render_utils.appearance_codebook_ref (arm_a/render_utils.py:32-61)
    /// Stratum by query aperture (closed ap<=0.20, wide ap>0.45, else mid),
    /// argmin L2 in standardized-6D against ref_geom6 within the stratum.
    /// Returns the codebook ROW (0..29).
    ///
    /// `previousRow`/`margin` add the retrieval hysteresis shipped on web in
    /// 4349fa3. Selection is otherwise MEMORYLESS, so serve changes exemplar on
    /// ~40% of consecutive frames — a real flicker source the offline harness
    /// never saw because it pins one reference per frame. `margin = 0` (the
    /// default) is bit-identical to memoryless selection, so every existing
    /// caller and every frozen reference render is unaffected.
    ///
    /// Port note: this mirrors web/src/math/reference.ts appearanceCodebookRow
    /// including the deliberate choice NOT to restrict the held row to the
    /// current pool. A within-pool-only variant was measured and is strictly
    /// worse on both axes (retrieval cost 1.0104 vs 0.9657, switching 0.336 vs
    /// 0.254 at the same margin), because holding across a pool edge is where
    /// most of the win comes from (pool-driven switches 1019 -> 567).
    static func appearanceCodebookRef(query pred6: [Float],
                                      aperture ap: Float,
                                      codebook: Serve320Codebook,
                                      refGeom6: [Float],
                                      previousRow: Int = -1,
                                      margin: Float = 0) -> Int {
        let stratum = ap <= 0.20 ? "closed" : (ap > 0.45 ? "wide" : "mid")
        var pool = codebook.rows(forStratum: stratum)
        if pool.isEmpty {
            pool = Array(0..<30)          // render_utils.py:59 fallback to allcb
        }
        func squaredDistance(_ row: Int) -> Float {
            var d: Float = 0
            for k in 0..<6 {
                let diff = refGeom6[row * 6 + k] - pred6[k]
                d += diff * diff
            }
            return d
        }
        var best = pool[0]
        var bestDist = Float.greatestFiniteMagnitude
        for row in pool {
            let d = squaredDistance(row)
            if d < bestDist {
                bestDist = d
                best = row
            }
        }
        // Hold the previous exemplar unless the pool's best beats it by the margin.
        if margin > 0, previousRow >= 0, previousRow != best {
            if !(bestDist * (1 + margin) < squaredDistance(previousRow)) { return previousRow }
        }
        return best
    }

    // MARK: models.landmark_heatmaps (arm_a/models.py:115-124)
    /// 20 gaussian heatmaps, sigma 4.0, written into `dst` as channel `c0..c0+19`
    /// of a (C,res,res) planar fp32 buffer with row stride res*res.
    static func landmarkHeatmaps(points: [SIMD2<Float>], res: Int, sigma: Float,
                                 dst: UnsafeMutablePointer<Float>, channelOffset c0: Int) {
        landmarkHeatmaps(
            points: points,
            width: res,
            height: res,
            sigma: sigma,
            dst: dst,
            channelOffset: c0)
    }

    /// Rectangular variant used by native-ROI renderer graphs. Coordinates are
    /// local to the supplied width/height; the square wrapper above preserves
    /// the historical 320 implementation and call sites.
    static func landmarkHeatmaps(points: [SIMD2<Float>], width: Int, height: Int,
                                 sigma: Float,
                                 dst: UnsafeMutablePointer<Float>, channelOffset c0: Int) {
        let inv = 1 / (2 * sigma * sigma)
        let plane = width * height
        var gaussianX = [Float](repeating: 0, count: width)
        var gaussianY = [Float](repeating: 0, count: height)
        for p in 0..<points.count {
            let base = dst + (c0 + p) * plane
            let cx = points[p].x, cy = points[p].y
            for x in 0..<width {
                let dx = Float(x) - cx
                gaussianX[x] = expf(-(dx * dx) * inv)
            }
            for y in 0..<height {
                let dy = Float(y) - cy
                gaussianY[y] = expf(-(dy * dy) * inv)
            }
            gaussianX.withUnsafeBufferPointer { xValues in
                gaussianY.withUnsafeBufferPointer { yValues in
                    guard let xBase = xValues.baseAddress,
                          let yBase = yValues.baseAddress else { return }
                    for y in 0..<height {
                        var rowScale = yBase[y]
                        vDSP_vsmul(xBase, 1, &rowScale,
                                   base + y * width, 1, vDSP_Length(width))
                    }
                }
            }
        }
    }

    // MARK: render_utils._umeyama_similarity (arm_a/render_utils.py:64-83)
    struct Similarity {
        var s: Float                 // scale
        var r: (Float, Float, Float, Float) // row-major 2x2 R
        var t: SIMD2<Float>          // translation
    }

    /// Least-squares similarity src -> dst (no reflection), fp32.
    /// 2x2 SVD via symmetric-eigen closed form; sign-consistent so R = U D Vt
    /// matches torch.linalg.svd + the reflection guard (render_utils.py:76-82).
    static func umeyamaSimilarity(src: [SIMD2<Float>], dst: [SIMD2<Float>]) -> Similarity {
        let k = Float(src.count)
        var muS = SIMD2<Float>.zero, muD = SIMD2<Float>.zero
        for i in 0..<src.count { muS += src[i]; muD += dst[i] }
        muS /= k; muD /= k
        var varS: Float = 0
        // cov = dst_c^T src_c / K (render_utils.py:75)
        var m00: Float = 0, m01: Float = 0, m10: Float = 0, m11: Float = 0
        for i in 0..<src.count {
            let sc = src[i] - muS, dc = dst[i] - muD
            varS += sc.x * sc.x + sc.y * sc.y
            m00 += dc.x * sc.x; m01 += dc.x * sc.y
            m10 += dc.y * sc.x; m11 += dc.y * sc.y
        }
        varS /= k
        m00 /= k; m01 /= k; m10 /= k; m11 /= k

        // SVD of cov = U S Vt via eigendecomposition of cov^T cov (symmetric 2x2).
        let a = m00 * m00 + m10 * m10          // col0 norm^2
        let b = m00 * m01 + m10 * m11          // cross
        let c = m01 * m01 + m11 * m11          // col1 norm^2
        let tr = a + c
        let det = a * c - b * b
        let disc = sqrt(max(0, tr * tr / 4 - det))
        var l1 = tr / 2 + disc                  // larger eigenvalue
        var l2 = tr / 2 - disc
        if l2 > l1 { swap(&l1, &l2) }
        // eigenvectors v1 (for l1), v2 (for l2), orthonormal
        var v1 = SIMD2<Float>(1, 0)
        if b != 0 {
            let cand = SIMD2<Float>(l1 - c, b)
            if simd_length(cand) > 1e-12 {
                v1 = simd_normalize(cand)
            } else {
                let alt = SIMD2<Float>(b, l1 - a)
                v1 = simd_length(alt) > 1e-12 ? simd_normalize(alt) : SIMD2<Float>(0, 1)
            }
        } else if c > a {
            v1 = SIMD2<Float>(0, 1)
        }
        let v2 = SIMD2<Float>(-v1.y, v1.x)
        let s1 = sqrt(max(l1, 0))
        let s2 = sqrt(max(l2, 0))
        // u_i = cov v_i / s_i
        func mulCov(_ v: SIMD2<Float>) -> SIMD2<Float> {
            SIMD2(m00 * v.x + m01 * v.y, m10 * v.x + m11 * v.y)
        }
        var u1 = s1 > 1e-12 ? mulCov(v1) / s1 : SIMD2<Float>(1, 0)
        if simd_length(u1) < 1e-6 { u1 = SIMD2<Float>(1, 0) }
        u1 = simd_normalize(u1)
        var u2 = s2 > 1e-12 ? mulCov(v2) / s2 : SIMD2<Float>(-u1.y, u1.x)
        if simd_length(u2) < 1e-6 { u2 = SIMD2<Float>(-u1.y, u1.x) }
        u2 = simd_normalize(u2)
        // d = sign(det(U Vt)) (render_utils.py:77)
        // det(U Vt) = det(U) det(V); det(V)=1 by construction (v2 = v1 rotated +90)
        let detU = u1.x * u2.y - u1.y * u2.x
        let d: Float = detU >= 0 ? 1 : -1
        // R = U D Vt with D = diag(1, d) (render_utils.py:78-80)
        // U = [u1 u2], D U-form: UD = [u1, d*u2]; R = (UD) Vt
        let ud1 = u1, ud2 = d * u2
        let r00 = ud1.x * v1.x + ud2.x * v2.x
        let r01 = ud1.x * v1.y + ud2.x * v2.y
        let r10 = ud1.y * v1.x + ud2.y * v2.x
        let r11 = ud1.y * v1.y + ud2.y * v2.y
        // s = (S1 + d*S2) / var_s (render_utils.py:81)
        let scale = (s1 + d * s2) / max(varS, 1e-8)
        // t = mu_d - s R mu_s (render_utils.py:82)
        let t = muD - scale * SIMD2<Float>(r00 * muS.x + r01 * muS.y,
                                           r10 * muS.x + r11 * muS.y)
        return Similarity(s: scale, r: (r00, r01, r10, r11), t: t)
    }

    // MARK: render_utils.align_ref (arm_a/render_utils.py:86-110)
    /// Similarity-warp the (UNMASKED) ref crop so its landmarks map onto the
    /// target's. refRGB01: planar (3,res,res) float [0,1]; output same layout.
    /// Sampling = grid_sample bilinear, padding border, align_corners=True
    /// (ref px = R^T(p - t)/s — render_utils.py:104-110).
    static func alignRef(refRGB01: [Float], res: Int,
                         refPts: [SIMD2<Float>], tgtPts: [SIMD2<Float>]) -> [Float] {
        let sim = umeyamaSimilarity(src: refPts, dst: tgtPts)   // fp32 warp math
        let invS = 1 / max(sim.s, 1e-8)
        // inverse map: ref = R^T (p - t) / s  (R orthonormal -> R^T = inverse)
        let (r00, r01, r10, r11) = sim.r
        var out = [Float](repeating: 0, count: 3 * res * res)
        let plane = res * res
        for y in 0..<res {
            for x in 0..<res {
                let px = Float(x) - sim.t.x
                let py = Float(y) - sim.t.y
                // R^T (p - t) / s
                let rx = (r00 * px + r10 * py) * invS
                let ry = (r01 * px + r11 * py) * invS
                // grid_sample bilinear, padding border, align_corners=True: the sample
                // location is clamped to the image extent FIRST (ATen compute_coordinates),
                // then taps are taken (render_utils.py:106-110).
                let rxc = min(max(rx, 0), Float(res - 1))
                let ryc = min(max(ry, 0), Float(res - 1))
                let x0 = rxc.rounded(.down), y0 = ryc.rounded(.down)
                let fx = rxc - x0, fy = ryc - y0
                let ix0 = Int(x0), iy0 = Int(y0)
                let ix1 = min(ix0 + 1, res - 1)
                let iy1 = min(iy0 + 1, res - 1)
                let i00 = iy0 * res + ix0, i01 = iy0 * res + ix1
                let i10 = iy1 * res + ix0, i11 = iy1 * res + ix1
                let w00 = (1 - fx) * (1 - fy), w01 = fx * (1 - fy)
                let w10 = (1 - fx) * fy, w11 = fx * fy
                let pix = y * res + x
                for ch in 0..<3 {
                    let b = ch * plane
                    out[b + pix] = refRGB01[b + i00] * w00 + refRGB01[b + i01] * w01
                        + refRGB01[b + i10] * w10 + refRGB01[b + i11] * w11
                }
            }
        }
        return out
    }
}
