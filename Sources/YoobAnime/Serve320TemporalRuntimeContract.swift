//
//  Serve320TemporalRuntimeContract.swift
//  Pure lifecycle contracts shared by the native-ROI runtime and its
//  deterministic command-line test. No Core ML or Metal dependency belongs
//  here: this is the authority for stale-stream rejection and truthful status.
//

import Foundation

struct Serve320TemporalBoostLease: Equatable, Sendable {
    fileprivate let revision: UInt64
}

struct Serve320TemporalBoostLifecycle: Sendable {
    private var revision: UInt64 = 0
    private var initializationAttempted = false

    var currentLease: Serve320TemporalBoostLease {
        Serve320TemporalBoostLease(revision: revision)
    }

    func accepts(_ lease: Serve320TemporalBoostLease) -> Bool {
        lease.revision == revision
    }

    /// Invalidate every pipeline created before this lifecycle boundary.
    /// Initialization remains one-shot: a missing shader/device must not be
    /// retried on every frame after a reset.
    @discardableResult
    mutating func reset() -> Serve320TemporalBoostLease {
        revision &+= 1
        return currentLease
    }

    /// Returns true exactly once for the lifetime of the owning session.
    mutating func beginInitializationAttempt() -> Bool {
        guard !initializationAttempted else { return false }
        initializationAttempted = true
        return true
    }
}

enum Serve320OptionalModelFallbackPolicy {
    static func shouldFallback(requestedIsOptional: Bool,
                               loadFailed: Bool) -> Bool {
        requestedIsOptional && loadFailed
    }
}

enum Serve320OptionalModelResolver {
    struct Selection<Value> {
        let value: Value
        let usedFallback: Bool
    }

    /// Resolve one optional candidate without coupling the fallback behavior to
    /// Core ML. The production loader and this file's standalone test therefore
    /// execute the same catch/fallback control flow.
    static func load<Value>(requestedID: String,
                            fallbackID: String,
                            requestedIsOptional: Bool,
                            loader: (String) async throws -> Value) async throws
        -> Selection<Value> {
        do {
            return Selection(
                value: try await loader(requestedID),
                usedFallback: false
            )
        } catch {
            guard Serve320OptionalModelFallbackPolicy.shouldFallback(
                requestedIsOptional: requestedIsOptional,
                loadFailed: true
            ) else {
                throw error
            }
            return Selection(
                value: try await loader(fallbackID),
                usedFallback: true
            )
        }
    }
}

struct Serve320TemporalBoostApplication: Equatable, Sendable {
    let handled: Bool
    let effectApplied: Bool

    static let unavailable = Serve320TemporalBoostApplication(
        handled: false,
        effectApplied: false
    )
    static let identity = Serve320TemporalBoostApplication(
        handled: true,
        effectApplied: false
    )
    static let effect = Serve320TemporalBoostApplication(
        handled: true,
        effectApplied: true
    )
}
