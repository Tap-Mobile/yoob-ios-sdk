import XCTest
@testable import YoobRealistic

final class LanczosResizeTests: XCTestCase {
    /// SplitMix64, so every run tests the same pixels and a failure reproduces.
    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }
    }

    /// The per-pixel resize shipped before the row-vectorized version (029b747), kept as the bit-exact reference.
    private static func reference(_ source: [UInt8], sourceSide: Int, targetSide: Int) -> [UInt8] {
        let taps = LanczosTaps.shared.taps(sourceSide: sourceSide, targetSide: targetSide)
        var horizontal = [Int32](repeating: 0, count: sourceSide * targetSide * 3)
        for y in 0..<sourceSide { for x in 0..<targetSide { for c in 0..<3 {
            var value = 0
            for k in 0..<8 { value += Int(source[(y * sourceSide + taps.indices[x * 8 + k]) * 3 + c]) * taps.weights[x * 8 + k] }
            horizontal[(y * targetSide + x) * 3 + c] = Int32(value)
        } } }
        var result = [UInt8](repeating: 0, count: targetSide * targetSide * 3)
        for y in 0..<targetSide { for x in 0..<targetSide { for c in 0..<3 {
            var value: Int64 = 0
            for k in 0..<8 { value += Int64(horizontal[(taps.indices[y * 8 + k] * targetSide + x) * 3 + c]) * Int64(taps.weights[y * 8 + k]) }
            result[(y * targetSide + x) * 3 + c] = UInt8(max(0, min(255, (value + (1 << 21)) >> 22)))
        } } }
        return result
    }

    func testEveryProductionCropSideMatchesTheReferenceExactly() {
        var generator = SeededGenerator(state: 0x1A2B3C4D)
        // The H08 host crop side is 333...347 (15 sides); the outer crop is always 304. Also a downscale and identity.
        for target in Array(333...347) + [256, 304] {
            let noise = (0..<(304 * 304 * 3)).map { _ in UInt8.random(in: 0...255, using: &generator) }
            XCTAssertEqual(AvatarCompositor.resizeLanczos4(noise, sourceSide: 304, targetSide: target),
                           Self.reference(noise, sourceSide: 304, targetSide: target), "seeded noise, target \(target)")
        }
        // Hard edges drive the Lanczos overshoot to both clamps.
        let bars = (0..<(304 * 304 * 3)).map { UInt8(($0 / 3 % 304) % 16 < 8 ? 255 : 0) }
        XCTAssertEqual(AvatarCompositor.resizeLanczos4(bars, sourceSide: 304, targetSide: 342),
                       Self.reference(bars, sourceSide: 304, targetSide: 342), "clamped edges")
    }

    func testSmallOddAndEdgeSizesMatchTheReferenceExactly() {
        var generator = SeededGenerator(state: 0x5EED)
        // 8 source sides x 9 target sides (72 cases): tap indices clamp at both borders, downscale and upscale.
        for sourceSide in [1, 2, 3, 7, 8, 9, 17, 31] {
            for target in [1, 2, 3, 5, 8, 11, 16, 33, 64] {
                let pixels = (0..<(sourceSide * sourceSide * 3)).map { _ in UInt8.random(in: 0...255, using: &generator) }
                XCTAssertEqual(AvatarCompositor.resizeLanczos4(pixels, sourceSide: sourceSide, targetSide: target),
                               Self.reference(pixels, sourceSide: sourceSide, targetSide: target), "\(sourceSide) -> \(target)")
            }
        }
    }

    func testAnEmptyTargetReturnsAnEmptyResultWithoutCrashing() {
        let pixels = [UInt8](repeating: 128, count: 304 * 304 * 3)
        XCTAssertEqual(AvatarCompositor.resizeLanczos4(pixels, sourceSide: 304, targetSide: 0), [])
    }
    func testTapCacheHoldsEveryProductionSide() {
        // Filling the cache with all 15 production sides must not evict any of them (previously capped at 9 entries). An
        // evicted table is recomputed identically, so check what is still cached, on a fresh cache that no other test has
        // filled.
        let cache = LanczosTaps()
        for target in 333...347 { _ = cache.taps(sourceSide: 304, targetSide: target) }
        for target in 333...347 { XCTAssertTrue(cache.isCached(sourceSide: 304, targetSide: target), "side \(target) was evicted") }
    }
}
