import Foundation
import CryptoKit

/// A signed description of one character version: which files make it up, how they split into chunks on the CDN, and
/// the order they download in. The SDK refuses any manifest whose signature does not verify against a pinned key.
public struct CharacterManifest: Codable, Sendable, Equatable {
    public struct Chunk: Codable, Sendable, Equatable {
        /// SHA-256 of the chunk's bytes (after transport decoding), lowercase hex. Also its CDN address.
        public let sha256: String
        public let size: Int
    }
    public struct File: Codable, Sendable, Equatable {
        /// Pack-relative path, forward slashes, no `.` or `..` components.
        public let path: String
        public let size: Int
        public let sha256: String
        /// 0 poster, 1 idle frames, 2 models, 3 remaining host data. Lower tiers download first.
        public let tier: Int
        public let chunks: [Chunk]
    }
    public struct Idle: Codable, Sendable, Equatable {
        public let frames: [String]
        public let fps: Double
    }
    public struct CalmHosts: Codable, Sendable, Equatable {
        public let first: Int, count: Int, framesPerHost: Int
        public let wideLast: Int?
    }
    public enum Engine: String, Codable, Sendable { case realistic, anime }

    public let schema: Int
    public let character: String
    public let version: String
    public let engine: Engine
    public let displayName: String
    public let width: Int
    public let height: Int
    public let poster: String
    public let idle: Idle
    public let calmHosts: CalmHosts?
    public let minSDK: String
    public let files: [File]

    public var totalBytes: Int { files.reduce(0) { $0 + $1.size } }

    func validate() throws {
        guard schema == 1 else { throw YoobError.unsupported("manifest schema \(schema)") }
        guard YoobVersion(minSDK).map({ $0 <= YoobVersion.current }) ?? false else {
            throw YoobError.unsupported("\(character) \(version) needs Yoob SDK \(minSDK) or newer")
        }
        guard (1...4096).contains(width), (1...4096).contains(height), idle.fps > 0, idle.fps <= 60 else {
            throw YoobError.invalidAssets("manifest geometry")
        }
        var seen = Set<String>()
        for file in files {
            try PackPath.check(file.path)
            guard seen.insert(file.path).inserted, file.size >= 0, Hex.isSHA256(file.sha256), (0...3).contains(file.tier),
                  file.chunks.reduce(0, { $0 + $1.size }) == file.size,
                  file.chunks.allSatisfy({ Hex.isSHA256($0.sha256) && $0.size > 0 && $0.size <= 16 << 20 }) else {
                throw YoobError.invalidAssets("manifest entry \(file.path)")
            }
        }
        for path in [poster] + idle.frames where !seen.contains(path) { throw YoobError.invalidAssets("missing \(path)") }
    }
}

/// The signed envelope the CDN serves: the exact manifest bytes, and an Ed25519 signature over them.
struct SignedManifest: Codable, Sendable {
    let keyId: String
    let payload: String
    let signature: String

    func open(keys: [String: Data]) throws -> (CharacterManifest, Data) {
        guard let raw = keys[keyId] else { throw YoobError.invalidAssets("unknown signing key \(keyId)") }
        guard let bytes = Data(base64Encoded: payload), let signature = Data(base64Encoded: signature),
              bytes.count <= 4 << 20 else { throw YoobError.invalidAssets("manifest encoding") }
        let key = try Curve25519.Signing.PublicKey(rawRepresentation: raw)
        guard key.isValidSignature(signature, for: bytes) else { throw YoobError.invalidAssets("manifest signature") }
        let manifest = try JSONDecoder().decode(CharacterManifest.self, from: bytes)
        try manifest.validate()
        return (manifest, bytes)
    }
}

/// Public keys whose signatures the SDK accepts, by key id. Rotation adds a key here before the CDN switches to it.
enum YoobSigningKeys {
    static let pinned: [String: Data] = [
        "yoob-2026-09": Data(base64Encoded: "QEpH/whI1TpREBfxZzdZG9JNZ9EY9UpQmWu0vqfpi/s=")!,
    ]
}

enum PackPath {
    static func check(_ relative: String) throws {
        guard !relative.isEmpty, relative.utf8.count <= 512, !relative.hasPrefix("/"), !relative.contains("\\"),
              relative.split(separator: "/", omittingEmptySubsequences: false)
                .allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw YoobError.invalidAssets("path \(relative)")
        }
    }
}

enum Hex {
    static func isSHA256(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }
    static func string(_ digest: some Sequence<UInt8>) -> String {
        digest.map { String(format: "%02x", $0) }.joined()
    }
}

struct YoobVersion: Comparable, Sendable {
    static let current = YoobVersion(Yoob.version)!
    let parts: [Int]
    init?(_ text: String) {
        let parts = text.split(separator: ".").map { Int($0) }
        guard parts.count == 3, parts.allSatisfy({ $0 != nil }) else { return nil }
        self.parts = parts.compactMap { $0 }
    }
    static func < (a: YoobVersion, b: YoobVersion) -> Bool { a.parts.lexicographicallyPrecedes(b.parts) }
}
