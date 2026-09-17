import XCTest
import CoreGraphics
@testable import YoobRealistic

final class HostFrameDecoderTests: XCTestCase {
    /// A 1-pixel-high image whose width identifies the host index it was decoded for.
    private static func image(for index: Int) -> CGImage {
        let context = CGContext(data: nil, width: index + 1, height: 1, bitsPerComponent: 8, bytesPerRow: (index + 1) * 4,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }

    private final class Loads: @unchecked Sendable {
        private let lock = NSLock()
        private var counts: [Int: Int] = [:]
        func record(_ index: Int) { lock.lock(); counts[index, default: 0] += 1; lock.unlock() }
        func count(_ index: Int) -> Int { lock.lock(); defer { lock.unlock() }; return counts[index] ?? 0 }
    }

    func testReturnsRequestedFrameAndDecodesPrefetchedFramesOnce() throws {
        let loads = Loads()
        let decoder = HostFrameDecoder { index in loads.record(index); return Self.image(for: index) }
        // Consecutive call frames: each request was prefetched by the previous one.
        for index in 0..<10 {
            XCTAssertEqual(try decoder.image(index, prefetch: [index + 1, index + 2]).width, index + 1)
        }
        for index in 0..<10 { XCTAssertEqual(loads.count(index), 1, "host \(index)") }
    }

    func testUnexpectedJumpStillDecodesTheRequestedFrame() throws {
        let decoder = HostFrameDecoder { Self.image(for: $0) }
        XCTAssertEqual(try decoder.image(3, prefetch: [4, 5]).width, 4)
        // A voice-first restart moves the call frame; the decoder must not return a stale prefetched frame.
        XCTAssertEqual(try decoder.image(40, prefetch: [41, 42]).width, 41)
        // Ping-pong turnaround at the end of the host clip.
        XCTAssertEqual(try decoder.image(374, prefetch: [373, 372]).width, 375)
        XCTAssertEqual(try decoder.image(373, prefetch: [372, 371]).width, 374)
    }

    func testDecodeErrorsReachTheCaller() {
        struct Broken: Error {}
        let decoder = HostFrameDecoder { index in
            if index == 2 { throw Broken() }
            return Self.image(for: index)
        }
        XCTAssertNoThrow(try decoder.image(1, prefetch: [2]))
        XCTAssertThrowsError(try decoder.image(2, prefetch: [3])) { XCTAssertTrue($0 is Broken) }
        // The failure is not cached forever once the frame falls out of the window.
        XCTAssertEqual(try decoder.image(3, prefetch: [4]).width, 4)
    }

    func testLanczosTapsAreReusedPerSidePair() {
        let first = LanczosTaps.shared.taps(sourceSide: 304, targetSide: 342)
        let second = LanczosTaps.shared.taps(sourceSide: 304, targetSide: 342)
        XCTAssertEqual(first.indices, second.indices)
        XCTAssertEqual(first.weights, second.weights)
        XCTAssertEqual(first.indices.count, 342 * 8)
        // Each row of 11-bit weights sums to about 2048 (unit gain).
        for row in 0..<342 { XCTAssertEqual(first.weights[(row * 8)..<(row * 8 + 8)].reduce(0, +), 2048, accuracy: 2) }
    }
}
