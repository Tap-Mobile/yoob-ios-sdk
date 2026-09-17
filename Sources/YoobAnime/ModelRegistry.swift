//
//  ModelRegistry.swift
//  Single manifest-driven CoreML model registry. models_manifest.json (bundled)
//  is the one place that knows every model's resource name, default compute
//  units, and load-bearing quirks (dynamic-shape ANE rejection, enum-shape
//  recompiles, boundary dtypes) — see the manifest notes before changing any
//  computeUnits default here or at a call site.
//
//  API: load(id:) honors the manifest computeUnits (call sites may override for
//  the documented env-var A/B knobs), release(id:) drops the cached instance
//  (RAM: optional heavy models can be released on return to video-only idle), and
//  footprintMB() reports id -> approx MB for the perf HUD.
//

import Foundation
import CoreML

struct ModelManifestEntry: Decodable {
    let id: String
    let resource: String
    let precision: String
    let computeUnits: String
    let ownerFeature: String
    let approxMB: Double?
    let notes: String
    let optional: Bool?

    enum CodingKeys: String, CodingKey {
        case id, resource, precision, computeUnits, notes, optional
        case ownerFeature = "owner_feature"
        case approxMB = "approx_mb"
    }

    var defaultComputeUnits: MLComputeUnits {
        switch computeUnits {
        case "cpuOnly": return .cpuOnly
        case "cpuAndGPU": return .cpuAndGPU
        case "cpuAndNeuralEngine": return .cpuAndNeuralEngine
        case "all": return .all
        default:
            assertionFailure("models_manifest.json: unknown computeUnits '\(computeUnits)' for \(id)")
            return .cpuAndNeuralEngine
        }
    }
}

enum ModelRegistryError: Error {
    case manifestMissing
    case unknownModel(String)
    case unknownResource(String)
    case resourceMissing(String)
}

actor ModelRegistry {
    static let shared = ModelRegistry()

    private struct Manifest: Decodable {
        let schemaVersion: Int
        let models: [ModelManifestEntry]
    }

    private var entriesByID: [String: ModelManifestEntry]?
    private var loaded: [String: (model: MLModel, units: MLComputeUnits)] = [:]
    private var inflight: [String: (task: Task<MLModel, Error>, units: MLComputeUnits)] = [:]

    // MARK: - manifest (parsed once)

    private func entries() throws -> [String: ModelManifestEntry] {
        if let entriesByID { return entriesByID }
        guard let url = YoobResources.url(forResource: "models_manifest", withExtension: "json") else {
            throw ModelRegistryError.manifestMissing
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
        let byID = Dictionary(uniqueKeysWithValues: manifest.models.map { ($0.id, $0) })
        entriesByID = byID
        return byID
    }

    func entry(_ id: String) throws -> ModelManifestEntry {
        guard let entry = try entries()[id] else {
            throw ModelRegistryError.unknownModel(id)
        }
        return entry
    }

    // MARK: - load / release / footprint

    /// Load (or return the cached) model for a manifest id, honoring the
    /// manifest computeUnits unless the caller passes an explicit override
    /// (env-var A/B knobs). Concurrent loads of the same id share one task;
    /// an override that differs from the cached units reloads and replaces.
    func load(_ id: String, computeUnits override: MLComputeUnits? = nil) async throws -> MLModel {
        let entry = try entry(id)
        let units = override ?? entry.defaultComputeUnits
        if let cached = loaded[id], cached.units == units { return cached.model }
        if let pending = inflight[id], pending.units == units {
            return try await pending.task.value
        }
        let task = Task { try await Self.loadModel(entry: entry, units: units) }
        inflight[id] = (task, units)
        do {
            let model = try await task.value
            inflight[id] = nil
            loaded[id] = (model, units)
            return model
        } catch {
            inflight[id] = nil
            throw error
        }
    }

    /// Deprecated-shim support: resolve a legacy resource base name to its
    /// manifest entry (Serve320Models.loadNamed callers).
    func load(resource: String, computeUnits override: MLComputeUnits? = nil) async throws -> MLModel {
        guard let entry = try entries().values.first(where: { $0.resource == resource }) else {
            throw ModelRegistryError.unknownResource(resource)
        }
        return try await load(entry.id, computeUnits: override)
    }

    /// Drop the registry's strong reference for `id`. The model deallocates
    /// once the last caller reference goes away; the next load(id:) reloads.
    func release(_ id: String) {
        loaded[id] = nil
    }

    /// id -> approx MB for every currently loaded model (perf-HUD footprint).
    /// Entries without a pinned approx_mb report 0.
    func footprintMB() -> [String: Double] {
        loaded.keys.reduce(into: [:]) { report, id in
            report[id] = (try? entry(id))?.approxMB ?? 0
        }
    }

    // MARK: - resolution + compile cache (pattern from Serve320Models/ModelRunner)

    private static func loadModel(entry: ModelManifestEntry, units: MLComputeUnits) async throws -> MLModel {
        let url = try await resolveURL(for: entry)
        let config = MLModelConfiguration()
        config.computeUnits = units
        config.allowLowPrecisionAccumulationOnGPU = true
        // Diagnostics-only: fastPrediction lowers the isolated model call but
        // regresses the real frame pipeline by contending with decode/composite.
        // Keep the switch for reproducible device A/B; production stays on the
        // default specialization unless a launch explicitly opts in.
        if #available(iOS 18.0, macOS 15.0, *),
           entry.id.hasPrefix("serve320.renderer"),
           ProcessInfo.processInfo.environment["AVATAR_SERVE320_FAST_PREDICTION"] == "1" {
            var hints = MLOptimizationHints()
            hints.specializationStrategy = .fastPrediction
            config.optimizationHints = hints
        }
        return try await MLModel.load(contentsOf: url, configuration: config)
    }

    /// Probe `.mlmodelc` first (pre-compiled wins — determinism, same toolchain
    /// as the device), then `.mlpackage` (compiled on first launch and cached).
    /// A resource may carry a subdirectory (SupertonicCoreML/<variant>/name).
    private static func resolveURL(for entry: ModelManifestEntry) async throws -> URL {
        let resource = entry.resource
        let subdirectory: String?
        let name: String
        if let slash = resource.lastIndex(of: "/") {
            subdirectory = String(resource[..<slash])
            name = String(resource[resource.index(after: slash)...])
        } else {
            subdirectory = nil
            name = resource
        }
        if let compiled = YoobResources.url(forResource: name, withExtension: "mlmodelc",
                                          subdirectory: subdirectory) {
            return compiled
        }
        if let package = YoobResources.url(forResource: name, withExtension: "mlpackage",
                                         subdirectory: subdirectory) {
            return try await compileBundledPackageIfNeeded(package, resourceName: name)
        }
        throw ModelRegistryError.resourceMissing(resource)
    }

    /// On-device compile with a Caches/CompiledCoreML cache (same key layout as
    /// the old Serve320Models cache so existing device caches stay valid).
    /// Compile lands in a staging dir and moves into place atomically — an
    /// interrupted compile never becomes the cached model (a mid-compile app
    /// kill once left a corrupt fresh-mil/stale-weights model that hung
    /// MLModel.load forever).
    private static func compileBundledPackageIfNeeded(_ packageURL: URL,
                                                      resourceName: String) async throws -> URL {
        let fm = FileManager.default
        let cacheRoot = try fm.url(for: .cachesDirectory, in: .userDomainMask,
                                   appropriateFor: nil, create: true)
            .appendingPathComponent("CompiledCoreML", isDirectory: true)
        try fm.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        let cachedURL = cacheRoot.appendingPathComponent("\(resourceName).mlmodelc",
                                                         isDirectory: true)
        if fm.fileExists(atPath: cachedURL.path) {
            return cachedURL
        }
        let compiledURL = try await MLModel.compileModel(at: packageURL)
        let staging = cacheRoot.appendingPathComponent("\(resourceName).staging.mlmodelc",
                                                       isDirectory: true)
        try? fm.removeItem(at: staging)
        try fm.copyItem(at: compiledURL, to: staging)
        do {
            try fm.moveItem(at: staging, to: cachedURL)
        } catch let error as CocoaError where error.code == .fileWriteFileExists {
            try? fm.removeItem(at: staging)   // another task won the race — fine
        }
        return cachedURL
    }
}
