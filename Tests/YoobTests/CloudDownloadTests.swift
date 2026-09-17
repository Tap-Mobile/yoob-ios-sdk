import XCTest
@testable import Yoob

/// Downloads both characters from the live CDN into a temporary store and opens them.
/// Set YOOB_DOWNLOAD_TOKEN to a download grant (yoob-cdn/tools/mint-grant.mjs).
final class CloudDownloadTests: XCTestCase {
    func testDownloadsVerifiesAndResumes() async throws {
        guard let token = ProcessInfo.processInfo.environment["YOOB_DOWNLOAD_TOKEN"] else { throw XCTSkip("no grant") }
        let credentials = YoobCredentials(sessionToken: "test", downloadToken: token)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("yoob-cloud-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AssetStore(root: root)
        for character in ["luna-realistic", "luna-anime"] {
            let manifest = try await store.manifest(character: character, version: nil, credentials: credentials)
            let clock = ContinuousClock(), start = clock.now
            let early = try await store.download(manifest, throughTier: 1, credentials: credentials) { _ in }
            let posterAt = start.duration(to: clock.now)
            XCTAssertTrue(FileManager.default.fileExists(atPath: early.appendingPathComponent(manifest.poster).path))
            // Interrupt the model download, then resume it.
            let partial = Task { try await store.download(manifest, throughTier: 3, credentials: credentials) { _ in } }
            try await Task.sleep(for: .milliseconds(700))
            partial.cancel()
            _ = try? await partial.value
            let directory = try await store.download(manifest, throughTier: 3, credentials: credentials) { _ in }
            let total = start.duration(to: clock.now)
            XCTAssertTrue(FileManager.default.fileExists(atPath: directory.appendingPathComponent(".complete").path))
            for file in manifest.files {
                let digest = try AssetStore.digest(of: directory.appendingPathComponent(file.path))
                XCTAssertEqual(digest, file.sha256, file.path)
            }
            print("\(character): poster+idle in \(posterAt), all \(manifest.totalBytes / 1_000_000) MB in \(total)")
            let engine = try await FaceEngines.load(manifest, root: directory)
            XCTAssertGreaterThan(engine.tailSamples, 0)
            let cached = await store.cachedManifest(character: character)
            XCTAssertEqual(cached?.version, manifest.version)
        }
    }
}
