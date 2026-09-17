import Foundation
import CryptoKit
import ImageIO

/// Pack-relative files whose checksum already matched their receipt.
final class VerifiedFiles: @unchecked Sendable {
    private let lock = NSLock()
    private var names: Set<String> = []
    func contains(_ name: String) -> Bool { lock.lock(); defer { lock.unlock() }; return names.contains(name) }
    func insert(_ name: String) { lock.lock(); names.insert(name); lock.unlock() }
}

public enum AvatarError: Error, LocalizedError {
    case invalidPack(String), unavailable, invalidAudio
    public var errorDescription: String? {
        switch self {
        case .invalidPack(let reason): "The companion assets could not be verified (\(reason))."
        case .unavailable: "The companion renderer is unavailable."
        case .invalidAudio: "The companion received an invalid audio window."
        }
    }
}

public struct AvatarManifest: Codable, Sendable {
    public struct HostFrame: Codable, Sendable {
        public let row: Int
        public let bbox: [Int]
        public let width: Int
        public let height: Int
        public let file: String
        /// Video frame index inside the shared container; present only in videoContainer packs, where it equals row.
        public let frame: Int?
    }
    public struct Receipt: Codable, Sendable { public let bytes: Int; public let sha256: String }
    public let version: Int
    public let identity: String
    public let fps: Int
    public let sampleRate: Int
    public let samplesPerFrame: Int
    public let encoderTailSamples: Int
    public let channelOrder: String
    public let innerSize: Int
    public let outerSize: Int
    public let outputSize: Int
    public let lookahead: Int
    public let leftContext: Int
    public let rightContext: Int
    public let bootstrap: Int
    public let waveformMean: Float
    public let waveformStd: Float
    public let sourceHostFrames: Int
    public let runtimeEncoder: String
    public let encoderWindowFrames: [Int]
    public let frames: [HostFrame]
    public let files: [String: Receipt]
    /// True when every host frame lives in one shared HEVC video (frame.file, decoded with VideoToolbox).
    public let videoContainer: Bool?
    /// Distance between keyframes in the shared video; seeks land on a multiple of this. Defaults to 15.
    public let videoKeyframeInterval: Int?
}

public struct AvatarPack: Sendable {
    public let root: URL
    public let manifest: AvatarManifest
    public let manifestHash: String
    public let innerPixels: Data
    public let outerPixels: Data
    public let closedAudio: [Float]
    /// Decoded host frames with read-ahead, shared by every copy of this pack.
    public let hostFrames: HostFrameDecoder
    let verified: VerifiedFiles
    public init(root: URL) throws {
        self.root = root.resolvingSymlinksInPath()
        let data = try Data(contentsOf: self.root.appendingPathComponent("manifest.json"))
        guard data.count <= 2_000_000 else { throw AvatarError.invalidPack("manifest size") }
        manifest = try JSONDecoder().decode(AvatarManifest.self, from: data)
        manifestHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        guard manifest.version == 1, manifest.fps == 25, manifest.sampleRate == 16000,
              manifest.samplesPerFrame == 640, manifest.encoderTailSamples == 80,
              manifest.channelOrder == "BGR", manifest.innerSize == 144, manifest.outerSize == 304,
              manifest.outputSize == 288, manifest.lookahead == 9, manifest.leftContext == 16,
              manifest.rightContext == 4, manifest.bootstrap == 8,
              manifest.waveformMean.isFinite, manifest.waveformStd.isFinite, manifest.waveformStd > 0.000001,
              manifest.frames.count == manifest.sourceHostFrames, (2...5250).contains(manifest.frames.count),
              manifest.encoderWindowFrames == [8] + Array(13...21) else { throw AvatarError.invalidPack("runtime contract") }
        for (index, frame) in manifest.frames.enumerated() {
            guard frame.row == index, frame.bbox.count == 4, (1...4096).contains(frame.width), (1...4096).contains(frame.height),
                  frame.bbox[0] >= 0, frame.bbox[1] >= 0, frame.bbox[2] <= frame.width, frame.bbox[3] <= frame.height,
                  frame.bbox[2] > frame.bbox[0], frame.bbox[3] > frame.bbox[1], frame.bbox[3] - frame.bbox[1] == frame.bbox[2] - frame.bbox[0],
                  manifest.files[frame.file] != nil else { throw AvatarError.invalidPack("host geometry") }
        }
        let receipts = manifest.files, assetRoot = self.root
        func load(_ relative: String, expected: Int) throws -> Data {
            guard let receipt = receipts[relative], receipt.bytes == expected else { throw AvatarError.invalidPack(relative) }
            let path = try Self.path(relative, root: assetRoot)
            let value = try Data(contentsOf: path, options: .mappedIfSafe)
            guard value.count == expected, SHA256.hash(data: value).map({ String(format: "%02x", $0) }).joined() == receipt.sha256 else { throw AvatarError.invalidPack(relative) }
            return value
        }
        innerPixels = try load("inner144.bgr", expected: manifest.frames.count * 144 * 144 * 3)
        outerPixels = try load("outer304.bgr", expected: manifest.frames.count * 304 * 304 * 3)
        let closed = try load("closed_audio.f32", expected: 40 * 1024 * 4)
        closedAudio = closed.withUnsafeBytes { raw in
            (0..<(40 * 1024)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        guard closedAudio.allSatisfy(\.isFinite) else { throw AvatarError.invalidPack("closed audio") }
        let verified = VerifiedFiles(), manifest = self.manifest, packRoot = self.root
        self.verified = verified
        if manifest.videoContainer == true {
            guard let videoFile = manifest.frames.first?.file, manifest.frames.allSatisfy({ $0.file == videoFile }),
                  manifest.frames.allSatisfy({ $0.frame == nil || $0.frame == $0.row }) else { throw AvatarError.invalidPack("host video") }
            let video = HostVideoDecoder(url: try Self.path(videoFile, root: packRoot),
                                         keyframeInterval: manifest.videoKeyframeInterval ?? 15, frameCount: manifest.frames.count,
                                         frameRate: manifest.fps)
            hostFrames = HostFrameDecoder { index in
                let host = manifest.frames[index]
                _ = try AvatarPack.verifiedURL(videoFile, root: packRoot, manifest: manifest, verified: verified)
                let image = try video.image(forVideoFrame: index)
                guard image.width == host.width, image.height == host.height else { throw AvatarError.invalidPack("host image") }
                return image
            }
            return
        }
        hostFrames = HostFrameDecoder { index in
            let host = manifest.frames[index]
            let url = try AvatarPack.verifiedURL(host.file, root: packRoot, manifest: manifest, verified: verified)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
                  image.width == host.width, image.height == host.height else { throw AvatarError.invalidPack("host image") }
            return image
        }
    }
    public static func path(_ relative: String, root: URL) throws -> URL {
        guard !relative.isEmpty, !relative.hasPrefix("/"), !relative.contains("\\"),
              relative.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw AvatarError.invalidPack("path") }
        let resolved = root.appendingPathComponent(relative).resolvingSymlinksInPath()
        guard resolved.path.hasPrefix(root.resolvingSymlinksInPath().path + "/") else { throw AvatarError.invalidPack("path escape") }
        return resolved
    }
    public func verifiedURL(_ relative: String) throws -> URL {
        try Self.verifiedURL(relative, root: root, manifest: manifest, verified: verified)
    }
    /// Checksums each file once per loaded pack; later calls for the same file skip the hash. The files are
    /// read-only pack assets, so re-hashing a host frame on every rendered frame only cost time.
    static func verifiedURL(_ relative: String, root: URL, manifest: AvatarManifest, verified: VerifiedFiles) throws -> URL {
        let path = try Self.path(relative, root: root)
        guard let receipt = manifest.files[relative] else { throw AvatarError.invalidPack("missing receipt") }
        if verified.contains(relative) { return path }
        let stream = try FileHandle(forReadingFrom: path); defer { try? stream.close() }
        var hash = SHA256(), bytes = 0
        while let chunk = try stream.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk); bytes += chunk.count }
        guard bytes == receipt.bytes, hash.finalize().map({ String(format: "%02x", $0) }).joined() == receipt.sha256 else { throw AvatarError.invalidPack("checksum") }
        verified.insert(relative)
        return path
    }
    public func verifyModel(_ directory: String) throws -> URL {
        let names = manifest.files.keys.filter { $0.hasPrefix(directory + "/") }
        guard names.count >= 3 else { throw AvatarError.invalidPack("model receipt") }
        for name in names { _ = try verifiedURL(name) }
        return try Self.path(directory, root: root)
    }
    /// When set, call frames walk only this calm stretch of the host clip, forward and back, holding each host frame for
    /// `framesPerHost` call frames. The call screen's idle frames are rendered from the same stretch, so switching between
    /// idle and speech never jumps to a different head pose.
    public var calmHosts: CalmHostWindow?
    public struct CalmHostWindow: Sendable, Equatable {
        public let first: Int, count: Int, framesPerHost: Int
        /// With enough speech ahead, the streaming renderer may walk on up to this host at the clip's own speed, and walks
        /// back into the calm stretch before the speech ends (`HostWalker`). nil keeps speech inside the calm stretch.
        public let wideLast: Int?
        public var calmLast: Int { first + count - 1 }
        public init(first: Int, count: Int, framesPerHost: Int, wideLast: Int? = nil) {
            self.first = first; self.count = count; self.framesPerHost = max(1, framesPerHost); self.wideLast = wideLast
        }
    }
    public func hostIndex(for frame: Int) -> Int {
        if let calm = calmHosts, calm.count > 1, calm.first >= 0, calm.first + calm.count <= manifest.frames.count {
            let period = 2 * (calm.count - 1), position = (max(0, frame) / calm.framesPerHost) % period
            return calm.first + (position < calm.count ? position : period - position)
        }
        let n = manifest.frames.count, period = 2 * (n - 1)
        let position = (max(0, frame) + 1) % period
        return position < n ? position : period - position
    }
}
