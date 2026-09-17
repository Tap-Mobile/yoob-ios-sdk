import Foundation

/// Opens a character pack an app ships on disk. The pack must carry the signed manifest envelope the CDN serves, and
/// every file must match the checksum the manifest lists, exactly as for downloaded packs.
enum LocalPack {
    /// The signed envelope, saved unchanged from `https://cdn.yoob.com/v1/characters/<id>/<version>.json`.
    static let signedManifestName = "character.signed.json"
    /// The plain manifest that SDK development packs use. Accepted only by DEBUG builds that opt in.
    static let unsignedManifestName = "character.json"

    static func open(_ directory: URL, keys: [String: Data] = YoobSigningKeys.pinned) async throws -> CharacterManifest {
        let signed = directory.appendingPathComponent(signedManifestName)
        let manifest: CharacterManifest
        if let data = try? Data(contentsOf: signed) {
            let envelope: SignedManifest
            do { envelope = try JSONDecoder().decode(SignedManifest.self, from: data) }
            catch { throw YoobError.invalidAssets("\(signedManifestName) is not a signed manifest envelope") }
            manifest = try envelope.open(keys: keys).0
        } else if let unsigned = try unsignedDevelopmentManifest(directory) {
            return unsigned
        } else {
            throw YoobError.invalidAssets(
                "\(signedManifestName) missing in \(directory.lastPathComponent); local packs must carry the signed manifest")
        }
        try await verifyFiles(manifest, in: directory)
        return manifest
    }

    /// Hashes every file the manifest lists. Packs are tens of megabytes, so this takes well under a second.
    static func verifyFiles(_ manifest: CharacterManifest, in directory: URL) async throws {
        let root = directory.standardizedFileURL.resolvingSymlinksInPath()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for file in manifest.files {
                group.addTask(priority: .userInitiated) {
                    let url = root.appendingPathComponent(file.path).standardizedFileURL.resolvingSymlinksInPath()
                    // Refuse symlinks that lead out of the pack.
                    guard url.path.hasPrefix(root.path + "/") else { throw YoobError.invalidAssets("path \(file.path)") }
                    let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int
                    guard size == file.size else { throw YoobError.invalidAssets("\(file.path) is missing or the wrong size") }
                    guard try AssetStore.digest(of: url) == file.sha256 else {
                        throw YoobError.invalidAssets("\(file.path) failed verification")
                    }
                }
            }
            try await group.waitForAll()
        }
    }

    /// SDK development only: an unsigned `character.json`, when a DEBUG build sets `YOOB_ALLOW_UNSIGNED_PACKS=1`.
    /// Release builds don't contain this path.
    private static func unsignedDevelopmentManifest(_ directory: URL) throws -> CharacterManifest? {
        #if DEBUG
        guard ProcessInfo.processInfo.environment["YOOB_ALLOW_UNSIGNED_PACKS"] == "1",
              let data = try? Data(contentsOf: directory.appendingPathComponent(unsignedManifestName)) else { return nil }
        let manifest = try JSONDecoder().decode(CharacterManifest.self, from: data)
        try manifest.validate()
        return manifest
        #else
        return nil
        #endif
    }
}
