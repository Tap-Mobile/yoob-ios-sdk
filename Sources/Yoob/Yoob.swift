import Foundation

/// Yoob renders a talking character on the device from any speech audio.
public enum Yoob {
    public static let version = "0.4.0"

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
    /// The session can't continue (heartbeats kept failing, or Yoob ended it and a new one couldn't be opened), so the
    /// character stopped rendering. Call `prepare()` to start a new session.
    case sessionEnded(String)
    /// Yoob voice ended or refused the conversation. `code` is the WebSocket close code (for example 4009 when the
    /// session reached its time limit); `message` can be shown to the user.
    case voiceSession(code: Int, message: String)

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
        case .sessionEnded(let detail): "The Yoob session ended: \(detail)."
        case .voiceSession(_, let message): message
        }
    }
}

extension YoobError {
    /// What a Yoob voice relay close code means to the person using the app.
    static func voiceClosed(code: Int) -> YoobError {
        let message = switch code {
        case 1011: "The voice service disconnected. Start the conversation again."
        case 1013: "Voice is busy right now. Try again in a moment."
        case 4000: "The voice service refused this app's request. Update the app and try again."
        case 4001: "The voice session was refused. Start the conversation again."
        case 4002: "The voice session expired before it connected. Start the conversation again."
        case 4003: "This voice session was already used. Start the conversation again."
        case 4008: "This conversation reached its usage limit."
        case 4009: "This conversation reached its time limit."
        case 4010: "The conversation ended because it was idle for too long."
        case 4029: "Voice has reached its usage limit for now. Try again later."
        default: "The conversation disconnected (\(code))."
        }
        return .voiceSession(code: code, message: message)
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

    /// The same session with a download grant renewed by a heartbeat.
    func renewingGrant(_ downloadToken: String) -> YoobCredentials {
        YoobCredentials(sessionToken: sessionToken, downloadToken: downloadToken, heartbeatSeconds: heartbeatSeconds,
                        apiBase: apiBase, cdnBase: cdnBase)
    }

    public init(sessionToken: String, downloadToken: String, heartbeatSeconds: Int = 30,
                apiBase: URL = URL(string: "https://api2.yoob.com")!, cdnBase: URL = URL(string: "https://cdn.yoob.com")!) {
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
                  apiBase: try c.decodeIfPresent(URL.self, forKey: .apiBase) ?? URL(string: "https://api2.yoob.com")!,
                  cdnBase: try c.decodeIfPresent(URL.self, forKey: .cdnBase) ?? URL(string: "https://cdn.yoob.com")!)
    }
}

/// Where a character's files come from. Either way, the avatar opens a metered session with `credentials` and sends
/// heartbeats while it is prepared.
public enum YoobSource: Sendable {
    /// Download from the Yoob CDN with credentials from your backend.
    case cloud(character: String, credentials: @Sendable () async throws -> YoobCredentials)
    /// A pack your app ships: a directory with the signed manifest the CDN serves, saved as `character.signed.json`
    /// (from `https://cdn.yoob.com/v1/characters/<id>/<version>.json`), and the files it lists. The signature and every
    /// file's checksum are verified before use; unsigned or modified packs are refused. Files load from disk, but the
    /// session is still opened and metered through `credentials`.
    case local(URL, credentials: @Sendable () async throws -> YoobCredentials)
}
