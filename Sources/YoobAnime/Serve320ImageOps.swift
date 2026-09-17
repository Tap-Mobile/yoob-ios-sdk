//
//  Serve320ImageOps.swift
//  Minimal cv2-equivalent image ops used by the Serve320 QA9 compositor path.
//  Each op mirrors the exact OpenCV semantics the python serve relies on
//  (border modes, kernel construction, rounding conventions) so the on-device
//  composite matches the b200 reference. Cites point at the python call sites.
//

import Foundation

enum Serve320ImageOps {

    struct RectWindow {
        let x0: Int
        let y0: Int
        let x1: Int
        let y1: Int

        func clamped(w: Int, h: Int) -> RectWindow {
            RectWindow(x0: min(max(x0, 0), w - 1),
                       y0: min(max(y0, 0), h - 1),
                       x1: min(max(x1, 0), w - 1),
                       y1: min(max(y1, 0), h - 1))
        }

        func expanded(by radius: Int, w: Int, h: Int) -> RectWindow {
            RectWindow(x0: max(0, x0 - radius), y0: max(0, y0 - radius),
                       x1: min(w - 1, x1 + radius), y1: min(h - 1, y1 + radius))
        }
    }

    /// Below this many rows the GCD fan-out costs more than the work saved.
    /// A 320 -> 260x340 mouth-box resize is comfortably above it; the small
    /// 320x320 plane ops used elsewhere are not.
    static let parallelRowThreshold = 64

    // MARK: - multicore fan-out for the per-frame support ops

    /// The support chain (aperture support, skin mask, combine/finish) ran on
    /// ONE core while the render thread waited on the ANE — and at ~6.7 ms it
    /// outlives the ~3.5 ms inference window, so its tail serialized straight
    /// into every frame (53% of frame work in the 2026-08-04 device profile).
    /// These helpers fan the existing loops across cores. They are bit-exact
    /// by construction: chunks write disjoint index ranges, per-pixel
    /// arithmetic and its evaluation order are untouched, and merged
    /// reductions are order-free (min/max/integer sums).
    ///
    /// OFF by default. The fan-out cuts the support stage ~30% and won the
    /// cooled mean/p95 bench, but `concurrentPerform` completes at the pace of
    /// its SLOWEST chunk: under real-session load (call audio, video decode,
    /// UI) a chunk occasionally parks behind pool work and the owner saw FRM
    /// spike to ~22 ms. A steady 14 beats a faster average that stutters, so
    /// serial is the default; `--avatar-parallel-support` or
    /// AVATAR_PARALLEL_SUPPORT=1 opts in (a future bounded-worker scheduler
    /// can earn the default back). `--avatar-serial-support` still forces off.
    nonisolated(unsafe) static var parallelOpsEnabled: Bool = {
        let arguments = ProcessInfo.processInfo.arguments
        if arguments.contains("--avatar-serial-support") { return false }
        if arguments.contains("--avatar-parallel-support") { return true }
        return ProcessInfo.processInfo.environment["AVATAR_PARALLEL_SUPPORT"] == "1"
    }()

    /// Default 3, not core count: the support chain runs WHILE Core ML and
    /// the pipelined canvas composite hold cores of their own, and a 6-way
    /// fan-out measurably starved them on device (inference +0.45 ms, canvas
    /// +1.1 ms in the 2026-08-05 same-binary A/B) — eating the support win.
    /// Three chunks keep the fan-out on the free cores. Override with
    /// AVATAR_SUPPORT_CHUNKS for device tuning.
    static let parallelChunks: Int = {
        let raw = ProcessInfo.processInfo.environment["AVATAR_SUPPORT_CHUNKS"]
        let requested = Int(raw ?? "") ?? 3
        return max(2, min(ProcessInfo.processInfo.activeProcessorCount, requested))
    }()

    /// Splits the inclusive line range [start, end] into chunks and runs
    /// `body(chunkStart, chunkEnd)` concurrently. Falls back to one serial
    /// call when the range is too small to pay for the dispatch (~25 us).
    @inline(__always)
    static func fanOutLines(_ start: Int, _ end: Int, minLines: Int = 48,
                            _ body: (Int, Int) -> Void) {
        let count = end - start + 1
        guard count > 0 else { return }
        guard parallelOpsEnabled, count >= minLines else {
            body(start, end)
            return
        }
        let chunks = min(parallelChunks, max(1, count / (minLines / 2)))
        guard chunks > 1 else {
            body(start, end)
            return
        }
        let size = (count + chunks - 1) / chunks
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            let s = start + chunk * size
            let e = min(end, s + size - 1)
            if s <= e { body(s, e) }
        }
    }


    // MARK: - morphology (binary masks)

    /// Downward-only binary dilation: output(y,x)=max(src[y-d...y,x]).
    /// The last active source row is sufficient, replacing d full-image shifts
    /// in aperture-gated support with one vertical pass.
    static func dilateDownBinary(_ src: [Float], w: Int, h: Int, d: Int,
                                 window: RectWindow? = nil) -> [Float] {
        guard d > 0 else {
            guard let window else { return src }
            let win = window.clamped(w: w, h: h)
            var dst = [Float](repeating: 0, count: w * h)
            guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }
            for y in win.y0...win.y1 {
                let row = y * w
                for x in win.x0...win.x1 { dst[row + x] = src[row + x] }
            }
            return dst
        }
        let win = (window ?? RectWindow(x0: 0, y0: 0, x1: w - 1, y1: h - 1))
            .clamped(w: w, h: h)
        var dst = [Float](repeating: 0, count: w * h)
        guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }
        // Columns are independent, so they fan out across cores; each keeps
        // the exact single-pass scan.
        src.withUnsafeBufferPointer { source in
            dst.withUnsafeMutableBufferPointer { destination in
                let sp = source.baseAddress!
                let dp = destination.baseAddress!
                fanOutLines(win.x0, win.x1, minLines: 96) { cx0, cx1 in
                    for x in cx0...cx1 {
                        var lastActive = Int.min / 2
                        for y in 0..<h {
                            if sp[y * w + x] > 0 { lastActive = y }
                            if y >= win.y0, y <= win.y1, y - lastActive <= d {
                                dp[y * w + x] = 1
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    /// cv2.erode with the 5x5 MORPH_ELLIPSE structuring element
    /// (per_frame_support.py:47 — cv2.getStructuringElement(MORPH_ELLIPSE,(5,5)),
    /// corners are 0). Input float {0,1}; OOB treated as +inf (cv2 default
    /// morphology border) so border pixels erode over in-image taps only.
    static func erodeEllipse5x5(_ src: [Float], w: Int, h: Int,
                                window: RectWindow? = nil) -> [Float] {
        // The cv2 5x5 ellipse is exactly:
        // min(h3[y-2], h5[y-1], h5[y], h5[y+1], h3[y+2]), where h3/h5 are
        // horizontal radius-1/radius-2 minima. Inputs are binary, so changing
        // the comparison order is bit-identical while roughly halving work.
        let win = (window ?? RectWindow(x0: 0, y0: 0, x1: w - 1, y1: h - 1))
            .clamped(w: w, h: h)
        var dst = [Float](repeating: 0, count: w * h)
        guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }
        var h3 = [Float](repeating: 0, count: w * h)
        var h5 = [Float](repeating: 0, count: w * h)
        let rowY0 = max(0, win.y0 - 2), rowY1 = min(h - 1, win.y1 + 2)
        let hx0 = max(0, win.x0 - 1), hx1 = min(w - 1, win.x1 + 1)
        // Both passes are row-independent (the vertical pass only reads rows
        // the horizontal pass fully wrote), so each fans out across cores.
        src.withUnsafeBufferPointer { source in
            h3.withUnsafeMutableBufferPointer { h3buf in
                h5.withUnsafeMutableBufferPointer { h5buf in
                    let sp = source.baseAddress!
                    let p3 = h3buf.baseAddress!
                    let p5 = h5buf.baseAddress!
                    fanOutLines(rowY0, rowY1) { cy0, cy1 in
                        for y in cy0...cy1 {
                            let row = y * w
                            for x in hx0...hx1 {
                                var m = sp[row + x]
                                if x > 0 { m = min(m, sp[row + x - 1]) }
                                if x < w - 1 { m = min(m, sp[row + x + 1]) }
                                p3[row + x] = m
                            }
                            for x in win.x0...win.x1 {
                                var m = p3[row + x]
                                if x > 0 { m = min(m, p3[row + x - 1]) }
                                if x < w - 1 { m = min(m, p3[row + x + 1]) }
                                p5[row + x] = m
                            }
                        }
                    }
                }
            }
        }
        h3.withUnsafeBufferPointer { h3buf in
            h5.withUnsafeBufferPointer { h5buf in
                dst.withUnsafeMutableBufferPointer { destination in
                    let p3 = h3buf.baseAddress!
                    let p5 = h5buf.baseAddress!
                    let dp = destination.baseAddress!
                    fanOutLines(win.y0, win.y1) { cy0, cy1 in
                        for y in cy0...cy1 {
                            let row = y * w
                            for x in win.x0...win.x1 {
                                var m = Float.greatestFiniteMagnitude
                                if y > 1 { m = min(m, p3[row - 2 * w + x]) }
                                if y > 0 { m = min(m, p5[row - w + x]) }
                                m = min(m, p5[row + x])
                                if y < h - 1 { m = min(m, p5[row + w + x]) }
                                if y < h - 2 { m = min(m, p3[row + 2 * w + x]) }
                                if m != Float.greatestFiniteMagnitude {
                                    dp[row + x] = m
                                }
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    /// cv2.morphologyEx(MORPH_CLOSE, rect kxk) on a binary float mask = dilate then
    /// erode (render_utils.host_skin_mask:423). OOB ignored (cv2 default border).
    static func morphCloseRect(_ src: [Float], w: Int, h: Int, k: Int,
                               window: RectWindow? = nil) -> [Float] {
        guard let window else {
            return rectExtrema(rectExtrema(src, w: w, h: h, k: k, isDilate: true),
                               w: w, h: h, k: k, isDilate: false)
        }
        let win = window.clamped(w: w, h: h)
        let dilated = rectExtrema(src, w: w, h: h, k: k, isDilate: true,
                                  window: win.expanded(by: k / 2, w: w, h: h))
        return rectExtrema(dilated, w: w, h: h, k: k, isDilate: false,
                           window: win)
    }

    /// cv2.dilate with rect kxk on binary float (render_utils.py:568, mouth exempt dilate).
    static func dilateRect(_ src: [Float], w: Int, h: Int, k: Int,
                           window: RectWindow? = nil) -> [Float] {
        rectExtrema(src, w: w, h: h, k: k, isDilate: true, window: window)
    }

    /// Rectangular binary erosion used to keep the final mouth patch away from
    /// the host face silhouette. The narrow host boundary carries the sharp
    /// anime jaw stroke; repainting it with renderer skin breaks that contour.
    static func erodeRect(_ src: [Float], w: Int, h: Int, k: Int,
                          window: RectWindow? = nil) -> [Float] {
        rectExtrema(src, w: w, h: h, k: k, isDilate: false, window: window)
    }

    /// Exact rectangular erosion for a binary {0,1} mask. A summed-area table
    /// makes the jaw guard O(w*h) instead of comparing 2*k taps per pixel.
    static func erodeRectBinary(_ src: [Float], w: Int, h: Int, k: Int,
                                window: RectWindow? = nil) -> [Float] {
        let r = k / 2
        let stride = w + 1
        let win = (window ?? RectWindow(x0: 0, y0: 0, x1: w - 1, y1: h - 1))
            .clamped(w: w, h: h)
        var dst = [Float](repeating: 0, count: w * h)
        guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }
        let ix0 = max(0, win.x0 - r), iy0 = max(0, win.y0 - r)
        let ix1 = min(w - 1, win.x1 + r), iy1 = min(h - 1, win.y1 + r)
        var integral = [Int](repeating: 0, count: (w + 1) * (h + 1))
        for y in iy0...iy1 {
            var rowSum = 0
            let srcRow = y * w
            let dstRow = (y + 1) * stride
            let prevRow = y * stride
            for x in ix0...ix1 {
                if src[srcRow + x] > 0 { rowSum += 1 }
                integral[dstRow + x + 1] = integral[prevRow + x + 1] + rowSum
            }
        }
        integral.withUnsafeBufferPointer { table in
            dst.withUnsafeMutableBufferPointer { destination in
                let ip = table.baseAddress!
                let dp = destination.baseAddress!
                fanOutLines(win.y0, win.y1) { cy0, cy1 in
                    for y in cy0...cy1 {
                        let y0 = max(0, y - r), y1 = min(h - 1, y + r)
                        for x in win.x0...win.x1 {
                            let x0 = max(0, x - r), x1 = min(w - 1, x + r)
                            let sum = ip[(y1 + 1) * stride + x1 + 1]
                                - ip[y0 * stride + x1 + 1]
                                - ip[(y1 + 1) * stride + x0]
                                + ip[y0 * stride + x0]
                            if sum == (x1 - x0 + 1) * (y1 - y0 + 1) {
                                dp[y * w + x] = 1
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    /// A rectangular max/min filter is separable. Two 1-D passes preserve the
    /// exact clipped-border result while replacing k*k comparisons per pixel
    /// with 2*k (k=13 for the live mouth exemption; k=5 for skin-mask close).
    private static func rectExtrema(_ src: [Float], w: Int, h: Int, k: Int,
                                    isDilate: Bool, window: RectWindow? = nil) -> [Float] {
        let r = k / 2
        let win = (window ?? RectWindow(x0: 0, y0: 0, x1: w - 1, y1: h - 1))
            .clamped(w: w, h: h)
        var tmp = [Float](repeating: 0, count: w * h)
        var dst = [Float](repeating: 0, count: w * h)
        guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }

        // Lines are independent; chunks fan out with one deque scratch each.
        func extremaLine(source: UnsafePointer<Float>, base: Int, stride: Int,
                         length: Int, outStart: Int, outEnd: Int,
                         deque: UnsafeMutablePointer<Int>,
                         destination: UnsafeMutablePointer<Float>) {
            var head = 0
            var tail = 0
            var loadedHigh = max(0, outStart - r) - 1
            for index in outStart...outEnd {
                let low = max(0, index - r)
                let high = min(length - 1, index + r)
                while loadedHigh < high {
                    loadedHigh += 1
                    let value = source[base + loadedHigh * stride]
                    while tail > head {
                        let lastValue = source[base + deque[tail - 1] * stride]
                        let shouldPop = isDilate ? value >= lastValue : value <= lastValue
                        if !shouldPop { break }
                        tail -= 1
                    }
                    deque[tail] = loadedHigh
                    tail += 1
                }
                while tail > head, deque[head] < low { head += 1 }
                destination[base + index * stride] = source[base + deque[head] * stride]
            }
        }

        let dequeCapacity = max(w, h) + 1
        src.withUnsafeBufferPointer { source in
            tmp.withUnsafeMutableBufferPointer { temp in
                let sp = source.baseAddress!
                let tp = temp.baseAddress!
                fanOutLines(max(0, win.y0 - r), min(h - 1, win.y1 + r)) { cy0, cy1 in
                    let deque = UnsafeMutablePointer<Int>.allocate(capacity: dequeCapacity)
                    defer { deque.deallocate() }
                    for y in cy0...cy1 {
                        extremaLine(source: sp, base: y * w, stride: 1, length: w,
                                    outStart: win.x0, outEnd: win.x1,
                                    deque: deque, destination: tp)
                    }
                }
            }
        }
        tmp.withUnsafeBufferPointer { temp in
            dst.withUnsafeMutableBufferPointer { destination in
                let tp = temp.baseAddress!
                let dp = destination.baseAddress!
                fanOutLines(win.x0, win.x1) { cx0, cx1 in
                    let deque = UnsafeMutablePointer<Int>.allocate(capacity: dequeCapacity)
                    defer { deque.deallocate() }
                    for x in cx0...cx1 {
                        extremaLine(source: tp, base: x, stride: w, length: h,
                                    outStart: win.y0, outEnd: win.y1,
                                    deque: deque, destination: dp)
                    }
                }
            }
        }
        return dst
    }

    // MARK: - Gaussian blur

    /// cv2.GaussianBlur(img, (0,0), sigmaX=sigma) for CV_32F input
    /// (per_frame_support.py:48): ksize = cvRound(sigma*4*2+1)|1, kernel
    /// k[i] = exp(-0.5*((i-c)/sigma)^2) normalized (cv2 getGaussianKernel),
    /// BORDER_REFLECT_101 (cv2 BORDER_DEFAULT), separable.
    static func gaussianBlur(_ src: [Float], w: Int, h: Int, sigma: Float,
                             window: RectWindow? = nil) -> [Float] {
        let ksize = (Int((sigma * 4 * 2 + 1).rounded()) | 1)
        let c = Float(ksize - 1) * 0.5
        var kernel = [Float](repeating: 0, count: ksize)
        var sum: Float = 0
        for i in 0..<ksize {
            let x = (Float(i) - c) / sigma
            kernel[i] = exp(-0.5 * x * x)
            sum += kernel[i]
        }
        for i in 0..<ksize { kernel[i] /= sum }
        func reflect(_ p: Int, _ n: Int) -> Int {
            if p < 0 { return -p }
            if p >= n { return 2 * (n - 1) - p }
            return p
        }
        let r = ksize / 2
        var tmp = [Float](repeating: 0, count: w * h)
        var dst = [Float](repeating: 0, count: w * h)
        let win = (window ?? RectWindow(x0: 0, y0: 0, x1: w - 1, y1: h - 1))
            .clamped(w: w, h: h)
        guard win.x0 <= win.x1, win.y0 <= win.y1 else { return dst }
        let tempY0 = max(0, min(win.y0 - r, 2 * (h - 1) - (win.y1 + r)))
        let tempY1 = min(h - 1, max(win.y1 + r, r - win.y0))
        // Rows are independent in both separable passes, so each fans out;
        // interior pixels (all taps in-bounds) skip the reflect() calls. Both
        // paths accumulate taps in the same ascending order, so results stay
        // bit-identical to the scalar version.
        src.withUnsafeBufferPointer { source in
            kernel.withUnsafeBufferPointer { kern in
                tmp.withUnsafeMutableBufferPointer { temp in
                    let sp = source.baseAddress!
                    let kp = kern.baseAddress!
                    let tp = temp.baseAddress!
                    let innerX0 = max(win.x0, r), innerX1 = min(win.x1, w - 1 - r)
                    fanOutLines(tempY0, tempY1) { cy0, cy1 in
                        for y in cy0...cy1 {
                            let row = y * w
                            var x = win.x0
                            while x <= win.x1, x < innerX0 || x > innerX1 {
                                var acc: Float = 0
                                for k in 0..<ksize {
                                    acc += sp[row + reflect(x + k - r, w)] * kp[k]
                                }
                                tp[row + x] = acc
                                x += 1
                            }
                            if innerX0 <= innerX1 {
                                while x <= min(win.x1, innerX1) {
                                    var acc: Float = 0
                                    let base = row + x - r
                                    for k in 0..<ksize {
                                        acc += sp[base + k] * kp[k]
                                    }
                                    tp[row + x] = acc
                                    x += 1
                                }
                            }
                            while x <= win.x1 {
                                var acc: Float = 0
                                for k in 0..<ksize {
                                    acc += sp[row + reflect(x + k - r, w)] * kp[k]
                                }
                                tp[row + x] = acc
                                x += 1
                            }
                        }
                    }
                }
            }
        }
        tmp.withUnsafeBufferPointer { temp in
            kernel.withUnsafeBufferPointer { kern in
                dst.withUnsafeMutableBufferPointer { destination in
                    let tp = temp.baseAddress!
                    let kp = kern.baseAddress!
                    let dp = destination.baseAddress!
                    fanOutLines(win.y0, win.y1) { cy0, cy1 in
                        for y in cy0...cy1 {
                            let interior = y - r >= 0 && y + r <= h - 1
                            if interior {
                                let base = (y - r) * w
                                for x in win.x0...win.x1 {
                                    var acc: Float = 0
                                    for k in 0..<ksize {
                                        acc += tp[base + k * w + x] * kp[k]
                                    }
                                    dp[y * w + x] = acc
                                }
                            } else {
                                for x in win.x0...win.x1 {
                                    var acc: Float = 0
                                    for k in 0..<ksize {
                                        acc += tp[reflect(y + k - r, h) * w + x] * kp[k]
                                    }
                                    dp[y * w + x] = acc
                                }
                            }
                        }
                    }
                }
            }
        }
        return dst
    }

    // MARK: - resize (cv2 INTER_LINEAR, half-pixel convention)

    /// cv2.resize INTER_LINEAR on float32: src = (dst+0.5)*scale - 0.5, taps
    /// clamped to the image extent (cv2 border-replicate). (canonical_composite.py:77)
    static func resizeBilinearFloat(_ src: [Float], sw: Int, sh: Int, dw: Int, dh: Int) -> [Float] {
        var dst = [Float](repeating: 0, count: dw * dh)
        let xTaps = axisTaps(source: sw, destination: dw)
        let yTaps = axisTaps(source: sh, destination: dh)
        src.withUnsafeBufferPointer { source in
            xTaps.index0.withUnsafeBufferPointer { xi0 in
            xTaps.index1.withUnsafeBufferPointer { xi1 in
            xTaps.weight.withUnsafeBufferPointer { xw in
                dst.withUnsafeMutableBufferPointer { destination in
                    // `destination` is inout and cannot be captured by the
                    // escaping concurrentPerform closure; the base address can.
                    let out0 = destination.baseAddress!
                    let body = { (y: Int) in
                        let iy0 = yTaps.index0[y], iy1 = yTaps.index1[y]
                        let wy = yTaps.weight[y]
                        let row0 = iy0 * sw, row1 = iy1 * sw
                        let out = y * dw
                        for x in 0..<dw {
                            let ix0 = xi0[x], ix1 = xi1[x], wx = xw[x]
                            let v00 = source[row0 + ix0], v01 = source[row0 + ix1]
                            let v10 = source[row1 + ix0], v11 = source[row1 + ix1]
                            out0[out + x] = (v00 * (1 - wx) + v01 * wx) * (1 - wy)
                                + (v10 * (1 - wx) + v11 * wx) * wy
                        }
                    }
                    if dh >= parallelRowThreshold {
                        DispatchQueue.concurrentPerform(iterations: dh, execute: body)
                    } else {
                        for y in 0..<dh { body(y) }
                    }
                }
            }}}
        }
        return dst
    }

    /// cv2.resize INTER_LINEAR on uint8 (canonical_composite.py:76 pred_box):
    /// same geometry; result rounded to u8 (cv2 fixed-point rounds half-up —
    /// within 1 LSB of the exact float computation, which is all the QA9 blend
    /// tolerance needs).
    /// Per-column bilinear coefficients. `fx`, `x0`, `wx`, `ix0` and `ix1`
    /// depend only on the column, but the original inner loop recomputed all of
    /// them for every pixel of every row — at a 260x340 box, ~88k redundant
    /// coefficient computations per channel per frame.
    struct AxisTaps {
        var index0: [Int]
        var index1: [Int]
        var weight: [Float]
    }

    static func axisTaps(source: Int, destination: Int) -> AxisTaps {
        let scale = Float(source) / Float(destination)
        var index0 = [Int](repeating: 0, count: destination)
        var index1 = [Int](repeating: 0, count: destination)
        var weight = [Float](repeating: 0, count: destination)
        for position in 0..<destination {
            let projected = (Float(position) + 0.5) * scale - 0.5
            var low = projected.rounded(.down)
            var fraction = projected - low
            if low < 0 { low = 0; fraction = 0 }
            if Int(low) >= source - 1 { low = Float(source - 1); fraction = 0 }
            let base = Int(low)
            index0[position] = base
            index1[position] = min(base + 1, source - 1)
            weight[position] = fraction
        }
        return AxisTaps(index0: index0, index1: index1, weight: weight)
    }

    /// Rows are independent and write disjoint destination ranges, so they fan
    /// out across cores. Arithmetic and evaluation order inside a pixel are
    /// unchanged, so output stays bit-identical to the scalar version.
    static func resizeBilinearU8(_ src: [UInt8], sw: Int, sh: Int, dw: Int, dh: Int) -> [UInt8] {
        var dst = [UInt8](repeating: 0, count: dw * dh)
        let xTaps = axisTaps(source: sw, destination: dw)
        let yTaps = axisTaps(source: sh, destination: dh)
        src.withUnsafeBufferPointer { source in
            xTaps.index0.withUnsafeBufferPointer { xi0 in
            xTaps.index1.withUnsafeBufferPointer { xi1 in
            xTaps.weight.withUnsafeBufferPointer { xw in
                dst.withUnsafeMutableBufferPointer { destination in
                    // `destination` is inout and cannot be captured by the
                    // escaping concurrentPerform closure; the base address can.
                    let out0 = destination.baseAddress!
                    let body = { (y: Int) in
                        let iy0 = yTaps.index0[y], iy1 = yTaps.index1[y]
                        let wy = yTaps.weight[y]
                        let row0 = iy0 * sw, row1 = iy1 * sw
                        let out = y * dw
                        for x in 0..<dw {
                            let ix0 = xi0[x], ix1 = xi1[x], wx = xw[x]
                            let v00 = Float(source[row0 + ix0]), v01 = Float(source[row0 + ix1])
                            let v10 = Float(source[row1 + ix0]), v11 = Float(source[row1 + ix1])
                            let v = (v00 * (1 - wx) + v01 * wx) * (1 - wy)
                                + (v10 * (1 - wx) + v11 * wx) * wy
                            out0[out + x] = UInt8(min(max((v + 0.5).rounded(.down), 0), 255))
                        }
                    }
                    if dh >= parallelRowThreshold {
                        DispatchQueue.concurrentPerform(iterations: dh, execute: body)
                    } else {
                        for y in 0..<dh { body(y) }
                    }
                }
            }}}
        }
        return dst
    }

    // MARK: - convex hull fill (render_utils.py:565-567 call site in replay.py)

    /// Andrew monotone chain over integer points -> hull vertices CCW (set ==
    /// cv2.convexHull output set; order may differ, fill is order-invariant).
    static func convexHull(_ pts: [SIMD2<Int>]) -> [SIMD2<Int>] {
        let sorted = pts.sorted { ($0.x, $0.y) < ($1.x, $1.y) }
        guard sorted.count > 2 else { return sorted }
        func cross(_ o: SIMD2<Int>, _ a: SIMD2<Int>, _ b: SIMD2<Int>) -> Int {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [SIMD2<Int>] = []
        for p in sorted {
            while lower.count >= 2,
                  cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 {
                lower.removeLast()
            }
            lower.append(p)
        }
        var upper: [SIMD2<Int>] = []
        for p in sorted.reversed() {
            while upper.count >= 2,
                  cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 {
                upper.removeLast()
            }
            upper.append(p)
        }
        lower.removeLast()
        upper.removeLast()
        return lower + upper
    }

    /// cv2.fillConvexPoly equivalent: binary (w,h) mask, 1 inside/on the hull.
    /// Fills pixel centers inside the convex hull (<=1px edge difference vs
    /// cv2's scanline fill; the 13x13 dilate downstream dominates).
    static func fillConvexPoly(_ hull: [SIMD2<Int>], w: Int, h: Int) -> [Float] {
        var dst = [Float](repeating: 0, count: w * h)
        guard hull.count >= 3 else { return dst }
        let n = hull.count
        // hull is CCW (monotone chain) -> inside = left of every edge
        let minY = max(0, hull.map { $0.y }.min()!)
        let maxY = min(h - 1, hull.map { $0.y }.max()!)
        guard minY <= maxY else { return dst }
        dst.withUnsafeMutableBufferPointer { destination in
            let dp = destination.baseAddress!
            fanOutLines(minY, maxY, minLines: 32) { cy0, cy1 in
                for y in cy0...cy1 {
                    let cy = Float(y) + 0.5
                    for x in 0..<w {
                        let cx = Float(x) + 0.5
                        var inside = true
                        for i in 0..<n {
                            let a = hull[i], b = hull[(i + 1) % n]
                            let cr = Float(b.x - a.x) * (cy - Float(a.y))
                                - Float(b.y - a.y) * (cx - Float(a.x))
                            if cr < 0 { inside = false; break }
                        }
                        if inside { dp[y * w + x] = 1 }
                    }
                }
            }
        }
        return dst
    }

    // MARK: - np.median (average of middle two on even counts)

    static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let s = values.sorted()
        let n = s.count
        if n % 2 == 1 { return s[n / 2] }
        return (s[n / 2 - 1] + s[n / 2]) / 2
    }

    /// Same np.median semantics via selection instead of a full sort. The
    /// skin-mask reference tone takes three medians; on the dark-host
    /// fallback each ran over the whole 102k-pixel crop, so three O(n log n)
    /// sorts per frame. A median is an order statistic — the selected values
    /// are identical whatever algorithm finds them — so this is bit-exact.
    /// Mutates `values` (partial reorder); callers pass scratch copies.
    static func medianInPlace(_ values: inout [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let n = values.count
        func select(_ k: Int) -> Float {
            var lo = 0, hi = n - 1
            values.withUnsafeMutableBufferPointer { buf in
                let p = buf.baseAddress!
                while lo < hi {
                    let pivot = p[k]
                    var i = lo, j = hi
                    repeat {
                        while p[i] < pivot { i += 1 }
                        while p[j] > pivot { j -= 1 }
                        if i <= j {
                            let t = p[i]; p[i] = p[j]; p[j] = t
                            i += 1; j -= 1
                        }
                    } while i <= j
                    if j < k { lo = i }
                    if k < i { hi = j }
                }
            }
            return values[k]
        }
        if n % 2 == 1 { return select(n / 2) }
        let upper = select(n / 2)
        // After selecting k, everything left of k is <= values[k]; the lower
        // middle is the maximum of that prefix.
        var lower = -Float.greatestFiniteMagnitude
        for i in 0..<(n / 2) { lower = max(lower, values[i]) }
        return (lower + upper) / 2
    }
}
