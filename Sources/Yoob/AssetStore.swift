import Foundation
import CryptoKit

/// Download progress, in bytes of verified content.
public struct YoobProgress: Sendable, Equatable {
    public let completedBytes: Int
    public let totalBytes: Int
    public var fraction: Double { totalBytes == 0 ? 0 : Double(completedBytes) / Double(totalBytes) }
}

/// Where downloads come from, and the grant that authorizes them. The grant is read before every request, so one a
/// heartbeat renews is used from the next request on.
final class CDNAccess: @unchecked Sendable {
    let cdnBase: URL
    private let lock = NSLock()
    private var token: String

    init(cdnBase: URL, downloadToken: String) { self.cdnBase = cdnBase; token = downloadToken }
    convenience init(_ credentials: YoobCredentials) {
        self.init(cdnBase: credentials.cdnBase, downloadToken: credentials.downloadToken)
    }

    var downloadToken: String {
        get { lock.withLock { token } }
        set { lock.withLock { token = newValue } }
    }
}

/// Downloads character packs from the CDN in content-addressed chunks, verifies every chunk and file, and keeps them in
/// Application Support (excluded from backup). An interrupted download resumes at the last verified chunk; a new version
/// reuses unchanged chunks that are still on disk.
actor AssetStore {
    static let shared = AssetStore()

    private let root: URL
    private let session: URLSession
    private var inUse: Set<String> = []
    /// Downloads in progress by destination path. A second request for the same file waits for the first.
    private var inFlight: [String: Task<Void, Error>] = [:]
    private static let parallelChunks = 6

    init(root: URL = URL.applicationSupportDirectory.appendingPathComponent("Yoob", isDirectory: true),
         session: URLSession? = nil) {
        self.root = root
        if let session { self.session = session } else {
            let configuration = URLSessionConfiguration.default
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            configuration.urlCache = nil
            configuration.waitsForConnectivity = true
            configuration.timeoutIntervalForResource = 600
            configuration.httpMaximumConnectionsPerHost = Self.parallelChunks
            self.session = URLSession(configuration: configuration)
        }
    }

    /// Fetches and verifies the signed manifest for the newest (or pinned) version of a character.
    func manifest(character: String, version: String?, credentials: YoobCredentials) async throws -> CharacterManifest {
        guard character.range(of: "^[a-z0-9][a-z0-9-]{0,63}$", options: .regularExpression) != nil else {
            throw YoobError.unsupported("character id \(character)")
        }
        let url = credentials.cdnBase.appendingPathComponent("v1/characters/\(character)/\(version ?? "latest").json")
        let token = credentials.downloadToken
        let data = try await fetch(url, token: { token }, limit: 4 << 20)
        let envelope: SignedManifest
        do { envelope = try JSONDecoder().decode(SignedManifest.self, from: data) }
        catch { throw YoobError.invalidAssets("manifest envelope") }
        let (manifest, bytes) = try envelope.open(keys: YoobSigningKeys.pinned)
        guard manifest.character == character, version == nil || manifest.version == version else {
            throw YoobError.invalidAssets("manifest identity")
        }
        let directory = packDirectory(manifest)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: directory.appendingPathComponent(".manifest.json"), options: .atomic)
        return manifest
    }

    /// The newest complete pack for a character already on disk, for starting offline.
    func cachedManifest(character: String) -> CharacterManifest? {
        let base = root.appendingPathComponent("characters/\(character)", isDirectory: true)
        let versions = (try? FileManager.default.contentsOfDirectory(atPath: base.path)) ?? []
        let decoded = versions.compactMap { version -> CharacterManifest? in
            let directory = base.appendingPathComponent(version, isDirectory: true)
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(".complete").path),
                  let bytes = try? Data(contentsOf: directory.appendingPathComponent(".manifest.json")),
                  let manifest = try? JSONDecoder().decode(CharacterManifest.self, from: bytes),
                  (try? manifest.validate()) != nil, manifest.character == character else { return nil }
            return manifest
        }
        return decoded.max { $0.version.compare($1.version, options: .numeric) == .orderedAscending }
    }

    func packDirectory(_ manifest: CharacterManifest) -> URL {
        root.appendingPathComponent("characters/\(manifest.character)/\(manifest.version)", isDirectory: true)
    }

    /// Downloads every file whose tier is at most `throughTier`. Files already verified are skipped.
    func download(_ manifest: CharacterManifest, throughTier: Int, credentials: YoobCredentials,
                  progress: @escaping @Sendable (YoobProgress) -> Void) async throws -> URL {
        try await download(manifest, throughTier: throughTier, access: CDNAccess(credentials), progress: progress)
    }

    func download(_ manifest: CharacterManifest, throughTier: Int, access: CDNAccess,
                  progress: @escaping @Sendable (YoobProgress) -> Void) async throws -> URL {
        let directory = packDirectory(manifest)
        inUse.insert(directory.path)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var rootURL = root; try? rootURL.setResourceValues(values)

        let wanted = manifest.files.filter { $0.tier <= throughTier }.sorted { ($0.tier, $0.path) < ($1.tier, $1.path) }
        let total = wanted.reduce(0) { $0 + $1.size }
        var done = 0
        var pending: [CharacterManifest.File] = []
        for file in wanted {
            if isVerified(file, in: directory) { done += file.size } else { pending.append(file) }
        }
        progress(YoobProgress(completedBytes: done, totalBytes: total))

        // Up to four files at a time; each file fetches its own chunks in parallel, and URLSession caps connections.
        let counter = ProgressCounter(done: done, total: total, report: progress)
        try await withThrowingTaskGroup(of: Void.self) { group in
            var queue = pending.makeIterator()
            func enqueue() {
                guard let file = queue.next() else { return }
                group.addTask {
                    try await self.download(file, into: directory, access: access) { bytes in
                        counter.add(bytes)
                    }
                }
            }
            for _ in 0..<4 { enqueue() }
            while try await group.next() != nil { enqueue() }
        }
        if manifest.files.allSatisfy({ isVerified($0, in: directory) }) {
            try Data().write(to: directory.appendingPathComponent(".complete"))
            try? prune(character: manifest.character, keeping: manifest.version)
        }
        return directory
    }

    private func download(_ file: CharacterManifest.File, into directory: URL, access: CDNAccess,
                          counted: @escaping @Sendable (Int) -> Void) async throws {
        let key = directory.appendingPathComponent(file.path).path
        if let running = inFlight[key] {
            try await running.value
            counted(file.size)
            return
        }
        let task = Task { try await self.fetchFile(file, into: directory, access: access, counted: counted) }
        inFlight[key] = task
        defer { inFlight[key] = nil }
        try await task.value
    }

    private func fetchFile(_ file: CharacterManifest.File, into directory: URL, access: CDNAccess,
                           counted: @Sendable (Int) -> Void) async throws {
        let destination = directory.appendingPathComponent(file.path)
        let partial = destination.appendingPathExtension("part")
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: partial.path) { FileManager.default.createFile(atPath: partial.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: partial)
        defer { try? handle.close() }

        // Chunks already written by an interrupted download are recorded next to the partial file.
        let ledgerURL = partial.appendingPathExtension("done")
        var finished = Set((try? String(contentsOf: ledgerURL, encoding: .utf8))?.split(separator: "\n").compactMap { Int($0) } ?? [])
        var offsets: [Int] = []
        var offset = 0
        for chunk in file.chunks { offsets.append(offset); offset += chunk.size }
        for index in finished where index < file.chunks.count { counted(file.chunks[index].size) }

        let missing = file.chunks.indices.filter { !finished.contains($0) }
        let session = session
        try await withThrowingTaskGroup(of: (Int, Data).self) { group in
            var queue = missing.makeIterator()
            func enqueue() {
                guard let index = queue.next() else { return }
                let chunk = file.chunks[index]
                let url = access.cdnBase.appendingPathComponent("v1/chunks/\(chunk.sha256)")
                group.addTask {
                    let data = try await Self.fetch(url, token: { access.downloadToken }, limit: chunk.size, session: session)
                    guard data.count == chunk.size, Hex.string(SHA256.hash(data: data)) == chunk.sha256 else {
                        throw YoobError.invalidAssets("chunk \(chunk.sha256.prefix(12))")
                    }
                    return (index, data)
                }
            }
            for _ in 0..<Self.parallelChunks { enqueue() }
            while let (index, data) = try await group.next() {
                try handle.seek(toOffset: UInt64(offsets[index]))
                try handle.write(contentsOf: data)
                finished.insert(index)
                try? finished.sorted().map(String.init).joined(separator: "\n").write(to: ledgerURL, atomically: true, encoding: .utf8)
                counted(data.count)
                enqueue()
            }
        }
        try handle.truncate(atOffset: UInt64(file.size))
        try handle.synchronize()
        guard try Self.digest(of: partial) == file.sha256 else {
            try? FileManager.default.removeItem(at: partial); try? FileManager.default.removeItem(at: ledgerURL)
            throw YoobError.invalidAssets(file.path)
        }
        try? FileManager.default.removeItem(at: destination)
        try FileManager.default.moveItem(at: partial, to: destination)
        try? FileManager.default.removeItem(at: ledgerURL)
        try Data((file.sha256 + "\n").utf8).write(to: stampURL(destination))
    }

    /// A file counts as verified when its stamp records the manifest hash and its size still matches. Full re-hashing
    /// happens when the engine loads the pack.
    private func isVerified(_ file: CharacterManifest.File, in directory: URL) -> Bool {
        let path = directory.appendingPathComponent(file.path)
        guard let stamp = try? String(contentsOf: stampURL(path), encoding: .utf8),
              stamp.trimmingCharacters(in: .whitespacesAndNewlines) == file.sha256,
              let size = (try? FileManager.default.attributesOfItem(atPath: path.path))?[.size] as? Int else { return false }
        return size == file.size
    }

    private func stampURL(_ file: URL) -> URL {
        file.deletingLastPathComponent().appendingPathComponent("." + file.lastPathComponent + ".sha256")
    }

    private func prune(character: String, keeping version: String) throws {
        let base = root.appendingPathComponent("characters/\(character)", isDirectory: true)
        for other in try FileManager.default.contentsOfDirectory(atPath: base.path) where other != version {
            let directory = base.appendingPathComponent(other, isDirectory: true)
            guard !inUse.contains(directory.path) else { continue }
            try? FileManager.default.removeItem(at: directory)
        }
    }

    func clear() throws {
        let base = root.appendingPathComponent("characters", isDirectory: true)
        guard let characters = try? FileManager.default.contentsOfDirectory(atPath: base.path) else { return }
        for character in characters {
            let folder = base.appendingPathComponent(character, isDirectory: true)
            for version in (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [] {
                let directory = folder.appendingPathComponent(version, isDirectory: true)
                if !inUse.contains(directory.path) { try? FileManager.default.removeItem(at: directory) }
            }
        }
    }

    func release(_ directory: URL) { inUse.remove(directory.path) }

    private func fetch(_ url: URL, token: @escaping @Sendable () -> String, limit: Int) async throws -> Data {
        try await Self.fetch(url, token: token, limit: limit, session: session)
    }

    /// `token` is read before every attempt, so a renewed grant is used on the next retry.
    static func fetch(_ url: URL, token: @Sendable () -> String, limit: Int, session: URLSession) async throws -> Data {
        var request = URLRequest(url: url)
        request.setValue("yoob-ios/\(Yoob.version)", forHTTPHeaderField: "X-Yoob-SDK")
        var lastError: Error?
        for attempt in 0..<4 {
            if attempt > 0 { try await Task.sleep(for: .milliseconds(400 << attempt)) }
            request.setValue("Bearer \(token())", forHTTPHeaderField: "Authorization")
            do {
                let (data, response) = try await session.data(for: request)
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                switch status {
                case 200: guard data.count <= limit else { throw YoobError.invalidAssets("oversized response") }; return data
                case 401, 403: throw YoobError.unauthorized
                case 404: throw YoobError.unsupported("not found on the CDN")
                case 402: throw YoobError.outOfCredit
                default: lastError = YoobError.network("HTTP \(status)")
                }
            } catch let error as YoobError {
                throw error
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                lastError = YoobError.network(error.localizedDescription)
            }
        }
        throw lastError ?? YoobError.network("unknown")
    }

    static func digest(of url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hash = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
        return Hex.string(hash.finalize())
    }
}

/// Thread-safe byte counter that reports progress as files finish chunks.
final class ProgressCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var done: Int
    private let total: Int
    private let report: @Sendable (YoobProgress) -> Void
    init(done: Int, total: Int, report: @escaping @Sendable (YoobProgress) -> Void) {
        self.done = done; self.total = total; self.report = report
    }
    func add(_ bytes: Int) {
        lock.lock(); done += bytes; let value = YoobProgress(completedBytes: done, totalBytes: total); lock.unlock()
        report(value)
    }
}
