import AVFoundation
import CoreGraphics
import VideoToolbox

/// Sequential VideoToolbox decode of the shared HEVC host-frame video of a videoContainer pack.
/// Call frames advance one host at a time with a short prefetch, so one forward cursor serves
/// nearly every request; a jump (renderer restart, ping-pong turnaround) reopens the reader at
/// the keyframe at or below the requested frame and decodes forward. Decodes are serialized
/// here and run on HostFrameDecoder's background queue, never on the render path; only the
/// delivered CGImages are retained, by HostFrameDecoder's bounded slots.
final class HostVideoDecoder: @unchecked Sendable {
    private let url: URL
    private let keyframeInterval: Int
    private let frameCount: Int
    private let frameRate: Int
    private let queue = DispatchQueue(label: "companion.host-video-decode")
    private var asset: AVAsset?
    private var track: AVAssetTrack?
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    /// Index of the video frame the reader delivers next; frameCount once the video is drained.
    private var nextIndex = -1

    init(url: URL, keyframeInterval: Int, frameCount: Int, frameRate: Int) {
        self.url = url
        self.keyframeInterval = max(1, keyframeInterval)
        self.frameCount = frameCount
        self.frameRate = frameRate
    }

    func image(forVideoFrame index: Int) throws -> CGImage {
        try queue.sync {
            if track == nil {
                let asset = AVAsset(url: url)
                guard let found = asset.tracks(withMediaType: .video).first else { throw AvatarError.invalidPack("host video") }
                self.asset = asset
                track = found
            }
            // Continue forward while the request is close ahead of the cursor; otherwise seek to
            // the keyframe at or below the request and decode forward from there.
            if nextIndex < 0 || index < nextIndex || index - nextIndex > keyframeInterval {
                try start(at: index - index % keyframeInterval)
            }
            while nextIndex <= index {
                guard let sample = output?.copyNextSampleBuffer() else { throw AvatarError.invalidPack("host video") }
                let shown = indexOf(sample)
                if shown == index {
                    guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw AvatarError.invalidPack("host video") }
                    var image: CGImage?
                    guard VTCreateCGImageFromCVPixelBuffer(buffer, options: nil, imageOut: &image) == noErr, let image else {
                        throw AvatarError.invalidPack("host video")
                    }
                    nextIndex = shown + 1
                    return image
                }
                nextIndex = shown + 1
            }
            throw AvatarError.invalidPack("host video")
        }
    }

    /// Lock-free: only called on the serial queue. Starts delivery at `keyframe`, which must be a keyframe index.
    private func start(at keyframe: Int) throws {
        guard let asset, let track else { throw AvatarError.invalidPack("host video") }
        let reader = try AVAssetReader(asset: asset)
        let start = CMTime(seconds: Double(keyframe) / Double(frameRate), preferredTimescale: 600)
        reader.timeRange = CMTimeRange(start: start, duration: .positiveInfinity)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        guard reader.canAdd(output) else { throw AvatarError.invalidPack("host video") }
        reader.add(output)
        guard reader.startReading() else { throw AvatarError.invalidPack("host video") }
        self.reader = reader
        self.output = output
        nextIndex = keyframe
    }

    private func indexOf(_ sample: CMSampleBuffer) -> Int {
        let seconds = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
        return min(frameCount - 1, max(0, Int((seconds * Double(frameRate)).rounded())))
    }
}
