import Foundation

/// Where the anime engine finds its downloaded asset pack. The Yoob facade sets `root` to the verified pack directory
/// before loading; lookups fall back to the host app's bundle, where apps that ship the pack inside the app keep it.
public enum YoobResources {
    nonisolated(unsafe) public static var root: URL?

    public static func url(forResource name: String, withExtension ext: String?, subdirectory: String? = nil) -> URL? {
        if let root {
            var candidate = root
            if let subdirectory { candidate.appendPathComponent(subdirectory, isDirectory: true) }
            candidate.appendPathComponent(ext.map { "\(name).\($0)" } ?? name)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            return nil
        }
        return Bundle.main.url(forResource: name, withExtension: ext, subdirectory: subdirectory)
    }

    public static func directory(_ name: String) -> URL? {
        if let root { return root.appendingPathComponent(name, isDirectory: true) }
        return Bundle.main.resourceURL?.appendingPathComponent(name, isDirectory: true)
    }
}
