import Foundation

/// Sends a session's heartbeats from the moment it opens, and decides when the session must stop.
///
/// Transient failures are retried with backoff. `maxFailures` failures in a row, a refused session (401, 403) or an
/// exhausted workspace (402, or `stop` with `out-of-credits`) end the session. A session the API no longer knows (404,
/// or `stop` for another reason, such as an app that slept in the background) is replaced through `renew`.
@MainActor
final class SessionHeartbeat {
    struct Hooks {
        /// The current session. Read before every beat, so a renewed session is picked up.
        var credentials: () -> YoobCredentials?
        /// Opens a new session through the app's backend. Throwing ends the session.
        var renew: () async throws -> Void
        /// A renewed download grant arrived.
        var grant: (_ token: String, _ expiresAt: String?) -> Void
        /// The session can't continue. Called once, after the heartbeat has stopped.
        var ended: (YoobError) -> Void
    }

    typealias Send = @Sendable (YoobCredentials) async -> SessionAPI.Outcome
    typealias Sleep = @Sendable (Duration) async throws -> Void

    let maxFailures: Int
    private(set) var isRunning = false
    private(set) var consecutiveFailures = 0
    private let hooks: Hooks
    private let send: Send
    private let sleep: Sleep
    private let retryDelay: (Int) -> Duration
    private var loop: Task<Void, Never>?
    private var inFlight: Task<Void, Never>?
    private var generation = 0

    init(hooks: Hooks, maxFailures: Int = 3,
         retryDelay: @escaping (Int) -> Duration = SessionHeartbeat.defaultRetryDelay,
         send: @escaping Send = { await SessionAPI.heartbeat($0) },
         sleep: @escaping Sleep = { try await Task.sleep(for: $0) }) {
        self.hooks = hooks; self.maxFailures = max(1, maxFailures)
        self.retryDelay = retryDelay; self.send = send; self.sleep = sleep
    }

    /// 2 s after the first failure, 6 s after the second, with jitter.
    nonisolated static func defaultRetryDelay(_ failures: Int) -> Duration {
        .milliseconds(Int(Double(failures <= 1 ? 2_000 : 6_000) * Double.random(in: 0.8...1.2)))
    }

    /// Starts beating every `heartbeatSeconds` of the current credentials. `automatic: false` only arms the heartbeat,
    /// for tests that call `beatNow()` themselves.
    func start(automatic: Bool = true) {
        guard !isRunning else { return }
        isRunning = true
        consecutiveFailures = 0
        generation += 1
        guard automatic else { return }
        let generation = generation
        loop = Task { [weak self, sleep] in
            while !Task.isCancelled {
                guard let seconds = self?.hooks.credentials()?.heartbeatSeconds else { return }
                do { try await sleep(.seconds(max(5, seconds))) } catch { return }
                guard let self, self.isRunning, self.generation == generation else { return }
                await self.beatNow()
            }
        }
    }

    func stop() {
        isRunning = false
        generation += 1
        loop?.cancel(); loop = nil
        inFlight?.cancel(); inFlight = nil
    }

    /// Beats now (for example when the app returns to the foreground) unless a beat is already running.
    func beatNow() async {
        guard isRunning else { return }
        if let inFlight { return await inFlight.value }
        let generation = generation
        let task = Task { await self.beat(generation) }
        inFlight = task
        await task.value
        if self.generation == generation { inFlight = nil }
    }

    private func beat(_ generation: Int) async {
        while isRunning, self.generation == generation {
            guard let credentials = hooks.credentials() else {
                return end(.unauthorized)
            }
            let outcome = await send(credentials)
            guard isRunning, self.generation == generation else { return }
            switch outcome {
            case .ok(let reply):
                consecutiveFailures = 0
                return await handle(reply)
            case .fatal(let error):
                return end(error)
            case .gone:
                return await renew()
            case .transient(let detail):
                consecutiveFailures += 1
                if consecutiveFailures >= maxFailures {
                    return end(.sessionEnded("the session could not be confirmed (\(detail))"))
                }
                do { try await sleep(retryDelay(consecutiveFailures)) } catch { return }
            }
        }
    }

    private func handle(_ reply: SessionAPI.Reply) async {
        if let grant = reply.grant, !grant.isEmpty { hooks.grant(grant, reply.grantExpiresAt) }
        guard reply.stop else { return }
        if reply.reason == "out-of-credits" { return end(.outOfCredit) }
        await renew()
    }

    private func renew() async {
        let generation = generation
        do {
            try await hooks.renew()
            if self.generation == generation { consecutiveFailures = 0 }
        } catch {
            guard self.generation == generation else { return }
            if case YoobError.outOfCredit = error { return end(.outOfCredit) }
            end(.sessionEnded("Yoob ended the session and a new one couldn't be opened"))
        }
    }

    private func end(_ error: YoobError) {
        guard isRunning else { return }
        stop()
        hooks.ended(error)
    }
}

enum SessionAPI {
    struct Reply: Decodable, Equatable {
        let stop: Bool
        let reason: String?
        /// A renewed download grant, when the current one is close to expiring.
        let grant: String?
        let grantExpiresAt: String?

        init(stop: Bool = false, reason: String? = nil, grant: String? = nil, grantExpiresAt: String? = nil) {
            self.stop = stop; self.reason = reason; self.grant = grant; self.grantExpiresAt = grantExpiresAt
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            stop = (try? c.decodeIfPresent(Bool.self, forKey: .stop)) ?? false
            reason = try? c.decodeIfPresent(String.self, forKey: .reason)
            grant = try? c.decodeIfPresent(String.self, forKey: .grant)
            if let text = try? c.decodeIfPresent(String.self, forKey: .grantExpiresAt) {
                grantExpiresAt = text
            } else if let seconds = try? c.decodeIfPresent(Double.self, forKey: .grantExpiresAt) {
                grantExpiresAt = String(Int(seconds))
            } else {
                grantExpiresAt = nil
            }
        }
        enum CodingKeys: String, CodingKey { case stop, reason, grant, grantExpiresAt = "grant_expires_at" }
    }

    enum Outcome: Equatable {
        case ok(Reply)
        /// The API no longer knows the session.
        case gone
        case fatal(YoobError)
        case transient(String)
    }

    static func heartbeat(_ credentials: YoobCredentials, session: URLSession = .shared) async -> Outcome {
        let data: Data, status: Int
        do {
            (data, status) = try await post("heartbeat", credentials: credentials, session: session)
        } catch {
            return .transient(error.localizedDescription)
        }
        return outcome(status: status, body: data)
    }

    static func outcome(status: Int, body: Data) -> Outcome {
        switch status {
        case 200...299:
            guard let reply = try? JSONDecoder().decode(Reply.self, from: body.isEmpty ? Data("{}".utf8) : body) else {
                return .transient("unreadable heartbeat reply")
            }
            return .ok(reply)
        case 401, 403: return .fatal(.unauthorized)
        case 402: return .fatal(.outOfCredit)
        case 404, 410: return .gone
        default: return .transient("HTTP \(status)")
        }
    }

    /// Ends the session. Best effort.
    static func end(_ credentials: YoobCredentials) async {
        _ = try? await post("end", credentials: credentials, session: .shared)
    }

    private static func post(_ action: String, credentials: YoobCredentials, session: URLSession) async throws -> (Data, Int) {
        var request = URLRequest(url: credentials.apiBase.appendingPathComponent("api/v1/sessions/\(action)"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(credentials.sessionToken)", forHTTPHeaderField: "Authorization")
        request.setValue("yoob-ios/\(Yoob.version)", forHTTPHeaderField: "X-Yoob-SDK")
        request.timeoutInterval = 10
        let (data, response) = try await session.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
}
