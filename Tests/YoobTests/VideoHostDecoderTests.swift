import AVFoundation
import XCTest
@testable import YoobRealistic

/// HostVideoDecoder against a synthetic 375-frame 64x64 HEVC video (keyframe every 15, like the
/// h08-v3 hosts.mp4): every frame carries an 8x8 bright block whose position encodes the frame
/// index, so a decoded frame is identified by block centroid, immune to encoder colour handling.
final class VideoHostDecoderTests: XCTestCase {
    private static let frameCount = 375
    private static let side = 128
    private static let keyframeInterval = 15
    private static let columns = 19
    private static let step = 6
    private static let block = 6

    private static func blockOrigin(for index: Int) -> (Int, Int) {
        ((index % columns) * step + 1, (index / columns) * step + 1)
    }

    private static func writeVideo(to url: URL) async throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: side,
            AVVideoHeightKey: side,
            AVVideoCompressionPropertiesKey: [
                AVVideoMaxKeyFrameIntervalKey: keyframeInterval,
                AVVideoQualityKey: 1.0,
            ],
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: side,
            kCVPixelBufferHeightKey as String: side,
        ])
        guard writer.canAdd(input) else { throw AvatarError.unavailable }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? AvatarError.unavailable }
        writer.startSession(atSourceTime: .zero)
        for index in 0..<frameCount {
            while !input.isReadyForMoreMediaData { try await Task.sleep(nanoseconds: 1_000_000) }
            var buffer: CVPixelBuffer?
            guard let pool = adaptor.pixelBufferPool, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess,
                  let buffer else { throw AvatarError.unavailable }
            CVPixelBufferLockBaseAddress(buffer, [])
            let (bx, by) = blockOrigin(for: index)
            let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<side {
                for x in 0..<side {
                    let bright = (bx..<(bx + block)).contains(x) && (by..<(by + block)).contains(y)
                    let pixel = base + y * stride + x * 4
                    pixel[0] = bright ? 240 : 30
                    pixel[1] = bright ? 240 : 30
                    pixel[2] = bright ? 240 : 30
                    pixel[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(buffer, [])
            guard adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(index), timescale: 25)) else {
                throw writer.error ?? AvatarError.unavailable
            }
        }
        input.markAsFinished()
        await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? AvatarError.unavailable }
    }

    /// Centre of the bright block in a decoded frame, in pixels.
    private static func blockCentre(of image: CGImage) throws -> (Double, Double) {
        guard image.width == side, image.height == side else { throw AvatarError.invalidPack("host image") }
        guard let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4,
                                      space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw AvatarError.unavailable }
        context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        guard let data = context.data else { throw AvatarError.unavailable }
        let pixels = data.assumingMemoryBound(to: UInt8.self)
        var sumX = 0.0, sumY = 0.0, count = 0.0
        for y in 0..<side {
            for x in 0..<side {
                let red = pixels[y * side * 4 + x * 4]
                if red > 128 {
                    sumX += Double(x) + 0.5
                    sumY += Double(y) + 0.5
                    count += 1
                }
            }
        }
        guard count > Double(block * block / 4) else { throw AvatarError.invalidPack("host video") }
        return (sumX / count, sumY / count)
    }

    private static func assertFrame(_ image: CGImage, is index: Int, file: StaticString = #filePath, line: UInt = #line) throws {
        let (cx, cy) = try blockCentre(of: image)
        let (bx, by) = blockOrigin(for: index)
        XCTAssertEqual(cx, Double(bx) + Double(block) / 2, accuracy: 1.5, "frame \(index) x", file: file, line: line)
        XCTAssertEqual(cy, Double(by) + Double(block) / 2, accuracy: 1.5, "frame \(index) y", file: file, line: line)
    }

    private func makeVideo() async throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("host-video-\(UUID().uuidString).mp4")
        try await Self.writeVideo(to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testDecodesAll375FramesSequentially() async throws {
        let decoder = HostVideoDecoder(url: try await makeVideo(), keyframeInterval: Self.keyframeInterval,
                                       frameCount: Self.frameCount, frameRate: 25)
        for index in 0..<Self.frameCount {
            try Self.assertFrame(try decoder.image(forVideoFrame: index), is: index)
        }
    }

    func testSeeksAndRestartsLandOnTheRequestedFrame() async throws {
        let decoder = HostVideoDecoder(url: try await makeVideo(), keyframeInterval: Self.keyframeInterval,
                                       frameCount: Self.frameCount, frameRate: 25)
        // Sequential inside one GOP, a mid-GOP start, a backward jump (renderer restart / ping-pong
        // turnaround), forward again, a long jump across many GOPs, and back to the first frame.
        for index in [100, 101, 102, 200, 40, 41, 374, 0, 7] {
            try Self.assertFrame(try decoder.image(forVideoFrame: index), is: index)
        }
    }

    func testFrameBeyondTheVideoThrows() async throws {
        let decoder = HostVideoDecoder(url: try await makeVideo(), keyframeInterval: Self.keyframeInterval,
                                       frameCount: Self.frameCount, frameRate: 25)
        XCTAssertThrowsError(try decoder.image(forVideoFrame: Self.frameCount))
    }

    func testMissingFileThrows() async throws {
        let decoder = HostVideoDecoder(url: FileManager.default.temporaryDirectory.appendingPathComponent("missing-\(UUID().uuidString).mp4"),
                                       keyframeInterval: Self.keyframeInterval, frameCount: Self.frameCount, frameRate: 25)
        XCTAssertThrowsError(try decoder.image(forVideoFrame: 0))
    }
}
