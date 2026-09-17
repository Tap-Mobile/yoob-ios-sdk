import Foundation
import CoreGraphics
import ImageIO

public struct AvatarImage: @unchecked Sendable {
    public let width: Int
    public let height: Int
    public let rgba: Data
    /// The space the pixel values are in: the host video's own space, so the display converts the whole frame, pasted
    /// face square included, the same way it converts the idle loop video.
    public var colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()
    public init(width: Int, height: Int, rgba: Data, colorSpace: CGColorSpace = CGColorSpaceCreateDeviceRGB()) {
        self.width = width; self.height = height; self.rgba = rgba; self.colorSpace = colorSpace
    }
    public func cgImage() throws -> CGImage {
        guard let provider = CGDataProvider(data: rgba as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { throw AvatarError.unavailable }
        return image
    }
}

public enum AvatarCompositor {
    /// Host PNG decode is the dominant per-frame cost (~16 ms of ~25 ms on an M-series Mac), so the next two
    /// host frames are decoded ahead on a background queue. Pixel math is unchanged and output is bit-exact.
    public static let prefetchFrames = 2

    /// `host` overrides the call frame's host (the streaming walker); `prefetchHosts` names the hosts to decode ahead.
    public static func compose(pack: AvatarPack, frame: Int, cropBGR: Data, host chosen: Int? = nil, prefetchHosts: [Int]? = nil) throws -> AvatarImage {
        guard cropBGR.count == 288 * 288 * 3 else { throw AvatarError.invalidPack("crop bytes") }
        let index = chosen.map { min(max(0, $0), pack.manifest.frames.count - 1) } ?? pack.hostIndex(for: frame), host = pack.manifest.frames[index]
        let next = prefetchHosts ?? (1...prefetchFrames).map { pack.hostIndex(for: frame + $0) }
        let image = try pack.hostFrames.image(index, prefetch: next)
        let width = host.width, height = host.height
        // Compose in the host picture's own RGB space, so drawing it converts nothing. The pack's crops were cut from the
        // same unconverted values; converting only the host (a decoded video frame is tagged BT.709, up to ~12 levels
        // brighter in skin tones on iOS) made the pasted face square visible.
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpaceCreateDeviceRGB()
        var output = Data(repeating: 255, count: width * height * 4)
        try output.withUnsafeMutableBytes { raw in
            let rgba = raw.bindMemory(to: UInt8.self)
            guard let context = CGContext(data: rgba.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { throw AvatarError.unavailable }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

            let outerOffset = index * 304 * 304 * 3
            var outer = [UInt8](pack.outerPixels[outerOffset..<(outerOffset + 304 * 304 * 3)])
            // The 144-space hole is repeated exactly twice inside the 288 output.
            outer.withUnsafeMutableBufferPointer { outer in
                cropBGR.withUnsafeBytes { crop in
                    for y in 8..<268 {
                        let destination = ((y + 8) * 304 + 16) * 3, source = (y * 288 + 8) * 3
                        (outer.baseAddress! + destination).update(from: crop.bindMemory(to: UInt8.self).baseAddress! + source, count: 270 * 3)
                    }
                }
            }
            let side = host.bbox[2] - host.bbox[0]
            let up = resizeLanczos4(outer, sourceSide: 304, targetSide: side)
            up.withUnsafeBufferPointer { up in
                for y in 0..<side {
                    let row = ((host.bbox[1] + y) * width + host.bbox[0]) * 4
                    for x in 0..<side {
                        let edge = min(x + 1, side - x, y + 1, side - y)
                        let pixel = row + x * 4, sourcePixel = (y * side + x) * 3
                        if edge >= 8 {
                            // alpha == 1: the blend below reduces exactly to the resized value.
                            rgba[pixel] = up[sourcePixel + 2]; rgba[pixel + 1] = up[sourcePixel + 1]; rgba[pixel + 2] = up[sourcePixel]
                            continue
                        }
                        let alpha = min(1, Float(edge) / 8)
                        for rgb in 0..<3 {
                            let value = alpha * Float(up[sourcePixel + 2 - rgb]) + (1 - alpha) * Float(rgba[pixel + rgb])
                            rgba[pixel + rgb] = UInt8(max(0, min(255, value.rounded(.toNearestOrEven))))
                        }
                    }
                }
            }
            // Match the accepted delivery policy: blacken white side bars without stretching the host.
            func whiteColumn(_ x: Int) -> Bool {
                var sum = 0
                for y in 0..<height { let p = (y * width + x) * 4; sum += Int(rgba[p]) + Int(rgba[p + 1]) + Int(rgba[p + 2]) }
                return Double(sum) / Double(height * 3) > 240
            }
            var left = 0, right = 0
            while left < min(40, width) && whiteColumn(left) { left += 1 }
            while right < min(40, width) && whiteColumn(width - 1 - right) { right += 1 }
            for y in 0..<height {
                for x in 0..<left { let p = (y * width + x) * 4; rgba[p] = 0; rgba[p + 1] = 0; rgba[p + 2] = 0 }
                for x in (width - right)..<width { let p = (y * width + x) * 4; rgba[p] = 0; rgba[p + 1] = 0; rgba[p + 2] = 0 }
            }
        }
        return AvatarImage(width: width, height: height, rgba: output, colorSpace: space)
    }

    // Separable 8-tap Lanczos with 11-bit coefficients, matching OpenCV's uint8 resize convention.
    /// Horizontal pass: 8 unrolled taps per output pixel in Int32 (a tap sum stays below 255 × Σ|w| ≈ 7e5). Vertical pass:
    /// each output row adds 8 weighted source scanlines as whole rows, which vectorizes, then rounds, shifts and clamps.
    /// Integer sums don't depend on order, so the output is identical to the per-pixel reference (`AvatarTests`).
    static func resizeLanczos4(_ source: [UInt8], sourceSide: Int, targetSide: Int) -> [UInt8] {
        // An empty target stays an empty result, as before. (The compositor's side comes from a manifest face box, which
        // `AvatarPack` validates as non-empty, so it is never 0 in production.)
        guard targetSide > 0 else { return [] }
        // The loops read `source` through unchecked pointers: a short buffer must stop here, not read out of bounds.
        precondition(sourceSide > 0 && source.count >= sourceSide * sourceSide * 3,
                     "resizeLanczos4 needs a \(sourceSide)×\(sourceSide) RGB source, got \(source.count) bytes")
        let taps = LanczosTaps.shared.taps(sourceSide: sourceSide, targetSide: targetSide)
        let rowWidth = targetSide * 3
        var horizontal = [Int32](repeating: 0, count: sourceSide * rowWidth)
        var result = [UInt8](repeating: 0, count: targetSide * rowWidth)
        var accumulator = [Int64](repeating: 0, count: rowWidth)
        source.withUnsafeBufferPointer { source in
            taps.byteOffsets.withUnsafeBufferPointer { offsets in
                taps.weights32.withUnsafeBufferPointer { weights in
                    horizontal.withUnsafeMutableBufferPointer { horizontal in
                        for y in 0..<sourceSide {
                            let row = source.baseAddress! + y * sourceSide * 3, out = horizontal.baseAddress! + y * rowWidth
                            for x in 0..<targetSide {
                                let o = offsets.baseAddress! + x * 8, w = weights.baseAddress! + x * 8
                                for c in 0..<3 {
                                    let s = row + c
                                    var value = Int32(s[o[0]]) &* w[0]
                                    value &+= Int32(s[o[1]]) &* w[1]; value &+= Int32(s[o[2]]) &* w[2]; value &+= Int32(s[o[3]]) &* w[3]
                                    value &+= Int32(s[o[4]]) &* w[4]; value &+= Int32(s[o[5]]) &* w[5]; value &+= Int32(s[o[6]]) &* w[6]
                                    value &+= Int32(s[o[7]]) &* w[7]
                                    out[x * 3 + c] = value
                                }
                            }
                        }
                    }
                }
            }
        }
        taps.indices.withUnsafeBufferPointer { indices in
            taps.weights32.withUnsafeBufferPointer { weights in
                horizontal.withUnsafeBufferPointer { horizontal in
                    accumulator.withUnsafeMutableBufferPointer { sum in
                        result.withUnsafeMutableBufferPointer { result in
                            for y in 0..<targetSide {
                                sum.update(repeating: 0)
                                for k in 0..<8 {
                                    let line = horizontal.baseAddress! + indices[y * 8 + k] * rowWidth, weight = Int64(weights[y * 8 + k])
                                    for i in 0..<rowWidth { sum[i] &+= Int64(line[i]) &* weight }
                                }
                                let out = result.baseAddress! + y * rowWidth
                                for i in 0..<rowWidth { out[i] = UInt8(clamping: (sum[i] &+ (1 << 21)) >> 22) }
                            }
                        }
                    }
                }
            }
        }
        return result
    }
}

/// Lanczos taps depend only on the two sides. The H08 host crop side varies per frame (15 sides, 333...347), so cache
/// every side pair a pack uses; the bound only guards against unbounded growth.
final class LanczosTaps: @unchecked Sendable {
    static let shared = LanczosTaps()
    struct Table {
        let indices: [Int]
        let weights: [Int]
        /// `indices * 3`: tap byte offsets within an RGB row, and the weights as Int32, for the unrolled resize loops.
        let byteOffsets: [Int]
        let weights32: [Int32]
        init(indices: [Int], weights: [Int]) {
            self.indices = indices; self.weights = weights
            byteOffsets = indices.map { $0 * 3 }; weights32 = weights.map { Int32($0) }
        }
    }
    private let lock = NSLock()
    private var tables: [Int: Table] = [:]

    /// Whether this side pair's table is still cached (not evicted).
    func isCached(sourceSide: Int, targetSide: Int) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return tables[sourceSide << 16 | targetSide] != nil
    }

    func taps(sourceSide: Int, targetSide: Int) -> Table {
        let key = sourceSide << 16 | targetSide
        lock.lock(); defer { lock.unlock() }
        if let table = tables[key] { return table }
        var indices: [Int] = [], weights: [Int] = []
        indices.reserveCapacity(targetSide * 8); weights.reserveCapacity(targetSide * 8)
        for destination in 0..<targetSide {
            let coordinate = Float((Double(destination) + 0.5) * Double(sourceSide) / Double(targetSide) - 0.5)
            let center = Int(floor(coordinate)), fraction = Double(coordinate - Float(center))
            var row = (0..<8).map { i -> Double in
                let x = fraction - Double(i - 3)
                if abs(x) < 1e-12 { return 1 }
                if abs(x) >= 4 { return 0 }
                return sin(.pi * x) * sin(.pi * x / 4) / (.pi * .pi * x * x / 4)
            }
            let sum = row.reduce(0, +)
            row = row.map { $0 / sum }
            indices += (0..<8).map { min(sourceSide - 1, max(0, center + $0 - 3)) }
            weights += row.map { Int(($0 * 2048).rounded(.toNearestOrEven)) }
        }
        let table = Table(indices: indices, weights: weights)
        if tables.count >= 64 { tables.removeAll() }
        tables[key] = table
        return table
    }
}

/// Decoded host frames for the current frame and the next few, decoded concurrently off the render path.
/// Call frames advance one host frame at a time (ping-pong), so the next frames are known in advance.
public final class HostFrameDecoder: @unchecked Sendable {
    private final class Slot: @unchecked Sendable {
        let done = DispatchGroup()
        var result: Result<CGImage, Error>?
    }
    private let load: @Sendable (Int) throws -> CGImage
    private let queue = DispatchQueue(label: "companion.host-frame-decode", qos: .userInitiated, attributes: .concurrent)
    private let lock = NSLock()
    private var slots: [Int: Slot] = [:]

    init(load: @escaping @Sendable (Int) throws -> CGImage) { self.load = load }

    /// Returns host `index` decoded (waiting for an in-flight decode) and starts decoding `prefetch`.
    /// Keeps at most the requested frames, so memory stays at (1 + prefetch) decoded host bitmaps.
    public func image(_ index: Int, prefetch: [Int]) throws -> CGImage {
        lock.lock()
        let slot = slots[index] ?? start(index)
        for next in prefetch where slots[next] == nil { _ = start(next) }
        let keep = Set([index] + prefetch)
        slots = slots.filter { keep.contains($0.key) }
        lock.unlock()
        slot.done.wait()
        return try slot.result!.get()
    }
    /// Lock must be held.
    private func start(_ index: Int) -> Slot {
        let slot = Slot(), load = self.load
        slot.done.enter()
        queue.async {
            slot.result = Result { try load(index) }
            slot.done.leave()
        }
        slots[index] = slot
        return slot
    }
}
