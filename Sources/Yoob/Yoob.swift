import Foundation

/// Yoob renders a talking character on the device from any speech audio.
public enum Yoob {
    public static let version = "0.1.0"

    /// Removes every downloaded character except the versions currently loaded.
    public static func clearCache() async throws { try await AssetStore.shared.clear() }
}

public enum YoobError: Error, LocalizedError, Equatable {
    /// The credentials were refused, or expired.
    case unauthorized
    /// The workspace has no credit left; the console stopped the session.
    case outOfCredit
    /// The network failed while downloading. Calling `prepare()` again resumes where it stopped.
    case network(String)
    /// A downloaded file or manifest did not match its signature or checksum. It was discarded.
    case invalidAssets(String)
    /// This device, OS or SDK version cannot run the character.
    case unsupported(String)
    /// The audio passed to `speak` was not mono 16-bit PCM at a supported rate.
    case invalidAudio(String)
    /// The renderer stopped. Audio keeps playing; the idle face stays on screen.
    case renderer(String)
    /// The user hasn't allowed microphone access.
    case permissionDenied(String)

    public var errorDescription: String? {
        switch self {
        case .unauthorized: "Yoob refused the credentials. Fetch a new session token from your backend."
        case .outOfCredit: "This Yoob workspace is out of credit."
        case .network(let detail): "The character download was interrupted (\(detail)). Try again to resume."
        case .invalidAssets(let detail): "A character file failed verification (\(detail))."
        case .unsupported(let detail): "This character can't run here: \(detail)."
        case .invalidAudio(let detail): "Yoob can't use this audio: \(detail)."
        case .renderer(let detail): "The character renderer stopped: \(detail)."
        case .permissionDenied(let detail): detail
        }
    }
}

/// What your backend returns after calling `POST /api/v1/avatar/sessions` with your Yoob API key.
/// Never put the API key itself in an app.
public struct YoobCredentials: Sendable, Decodable, Equatable {
    /// Authorizes heartbeats and the end call for this one session only.
    public let sessionToken: String
    /// Short-lived grant the CDN checks before serving character files.
    public let downloadToken: String
    public let heartbeatSeconds: Int
    public let apiBase: URL
    public let cdnBase: URL

    public init(sessionToken: String, downloadToken: String, heartbeatSeconds: Int = 30,
                apiBase: URL = URL(string: "https://api.yoob.com")!, cdnBase: URL = URL(string: "https://cdn.yoob.com")!) {
        self.sessionToken = sessionToken; self.downloadToken = downloadToken
        self.heartbeatSeconds = max(5, heartbeatSeconds); self.apiBase = apiBase; self.cdnBase = cdnBase
    }

    enum CodingKeys: String, CodingKey {
        case sessionToken = "session_token", downloadToken = "download_token", heartbeatSeconds = "heartbeat_seconds"
        case apiBase = "api_base", cdnBase = "cdn_base"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(sessionToken: try c.decode(String.self, forKey: .sessionToken),
                  downloadToken: try c.decode(String.self, forKey: .downloadToken),
                  heartbeatSeconds: try c.decodeIfPresent(Int.self, forKey: .heartbeatSeconds) ?? 30,
                  apiBase: try c.decodeIfPresent(URL.self, forKey: .apiBase) ?? URL(string: "https://api.yoob.com")!,
                  cdnBase: try c.decodeIfPresent(URL.self, forKey: .cdnBase) ?? URL(string: "https://cdn.yoob.com")!)
    }
}

/// Where a character's files come from.
public enum YoobSource: Sendable {
    /// Download from the Yoob CDN with credentials from your backend. Sessions are metered.
    case cloud(character: String, credentials: @Sendable () async throws -> YoobCredentials)
    /// A pack already on disk (a manifest.json plus its files), for apps that ship the files themselves and for
    /// development. No network calls, no metering.
    case local(URL)
}
