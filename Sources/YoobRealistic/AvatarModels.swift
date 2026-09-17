import Foundation
import CryptoKit
@preconcurrency import CoreML

public actor AvatarModels {
    public let pack: AvatarPack
    private let encoder: MLModel
    private let renderer: MLModel
    private let steadyEncoder: MLModel
    private let encoderOutput: String
    private let rendererOutput: String
    private let steadyOutput: String
    /// `units` applies to the accelerated models (steady encoder and renderer). After an install `.all` specializes for the
    /// Neural Engine for ~72 s on the owner's iPhone Air; `.cpuAndGPU` loads in ~0.3 s but costs ~4x more per frame.
    public static func load(pack: AvatarPack, cpuOnly: Bool = false, units: MLComputeUnits = .all,
                            priority: TaskPriority = .userInitiated) async throws -> AvatarModels {
        try await Task.detached(priority: priority) { try AvatarModels(pack: pack, cpuOnly: cpuOnly, units: units) }.value
    }
    /// The GPU and Neural Engine loads start together after an install: each package compiles once, not twice at once.
    private static let compileLock = NSLock()
    public init(pack: AvatarPack, cpuOnly: Bool = false, units: MLComputeUnits = .all) throws {
        self.pack = pack
        func load(_ directory: String, units: MLComputeUnits) throws -> MLModel {
            let path = try pack.verifyModel(directory)
            let receipts = pack.manifest.files.keys.filter { $0.hasPrefix(directory + "/") }.sorted()
                .map { $0 + ":" + pack.manifest.files[$0]!.sha256 }.joined(separator: "\n")
            let key = SHA256.hash(data: Data(receipts.utf8)).map { String(format: "%02x", $0) }.joined()
            let folder = URL.cachesDirectory.appendingPathComponent("LanguageCompanions/CoreML/" + key)
            let compiled = folder.appendingPathComponent("model.mlmodelc")
            try Self.compileLock.withLock {
                if !FileManager.default.fileExists(atPath: compiled.path) {
                    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    let result = try MLModel.compileModel(at: path)
                    do { try FileManager.default.moveItem(at: result, to: compiled) }
                    catch { if !FileManager.default.fileExists(atPath: compiled.path) { throw error } }
                }
            }
            let configuration = MLModelConfiguration(); configuration.computeUnits = units
            return try MLModel(contentsOf: compiled, configuration: configuration)
        }
        encoder = try load(pack.manifest.runtimeEncoder, units: .cpuOnly)
        steadyEncoder = cpuOnly ? encoder : try load("encoder21.mlpackage", units: units)
        renderer = try load("renderer.mlpackage", units: cpuOnly ? .cpuOnly : units)
        guard encoder.modelDescription.inputDescriptionsByName["audio"] != nil,
              renderer.modelDescription.inputDescriptionsByName["image"] != nil,
              renderer.modelDescription.inputDescriptionsByName["audio"] != nil,
              encoder.modelDescription.outputDescriptionsByName.count == 1,
              renderer.modelDescription.outputDescriptionsByName.count == 1,
              let first = encoder.modelDescription.outputDescriptionsByName.keys.first,
              let second = renderer.modelDescription.outputDescriptionsByName.keys.first else { throw AvatarError.invalidPack("model interface") }
        guard let steady = steadyEncoder.modelDescription.outputDescriptionsByName.keys.first else { throw AvatarError.invalidPack("steady encoder") }
        encoderOutput = first; rendererOutput = second; steadyOutput = steady
    }
    /// One steady encode and one render on silence, so a call's first frame never pays Core ML's first-prediction
    /// specialization (~6.7 s on the GPU after an install, measured on the owner's iPhone Air).
    public func warmUp() throws {
        if pack.manifest.encoderWindowFrames.contains(21) { _ = try encode([Float](repeating: 0, count: 21 * 640 + 80), frameCount: 21) }
        _ = try renderCrop(frame: 0, audio: [Float](repeating: 0, count: 40 * 1024))
    }
    public func encode(_ samples: [Float], frameCount: Int) throws -> [Float] {
        guard pack.manifest.encoderWindowFrames.contains(frameCount), samples.count == frameCount * 640 + 80,
              samples.allSatisfy(\.isFinite) else { throw AvatarError.invalidAudio }
        let input = try MLMultiArray(shape: [1, NSNumber(value: samples.count)], dataType: .float32)
        let pointer = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        for index in samples.indices { pointer[index] = (samples[index] - pack.manifest.waveformMean) / pack.manifest.waveformStd }
        let selected = frameCount == 21 ? steadyEncoder : encoder
        let result = try selected.prediction(from: MLDictionaryFeatureProvider(dictionary: ["audio": input]))
        guard let output = result.featureValue(for: frameCount == 21 ? steadyOutput : encoderOutput)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, frameCount * 2, 1024] else { throw AvatarError.invalidPack("encoder output") }
        return try Self.floats(output)
    }
    /// Returns BGR uint8 pixels at 288 square, matching the exported model's channel convention.
    /// `referenceHost` optionally replaces the unmasked reference channels (0–2) with another host frame.
    /// Diagnostic only (`AvatarModelProbe --variants`): swapping the reference barely changes the silent
    /// mouth (about one grey level), so production rendering always uses the current host frame.
    public func renderCrop(frame: Int, audio: [Float], referenceHost: Int? = nil, host chosen: Int? = nil) throws -> Data {
        guard audio.count == 40 * 1024, audio.allSatisfy(\.isFinite) else { throw AvatarError.invalidAudio }
        let host = chosen.map { min(max(0, $0), pack.manifest.frames.count - 1) } ?? pack.hostIndex(for: frame), plane = 144 * 144
        let reference = referenceHost.map { min(max(0, $0), pack.manifest.frames.count - 1) } ?? host
        let image = try MLMultiArray(shape: [1, 6, 144, 144], dataType: .float32)
        let pointer = image.dataPointer.bindMemory(to: Float.self, capacity: image.count)
        let offset = host * plane * 3, referenceOffset = reference * plane * 3
        for y in 0..<144 {
            for x in 0..<144 {
                let pixel = y * 144 + x, hole = (4..<139).contains(x) && (4..<134).contains(y)
                for channel in 0..<3 {
                    pointer[channel * plane + pixel] = Float(pack.innerPixels[referenceOffset + pixel * 3 + channel]) / 255
                    pointer[(channel + 3) * plane + pixel] = hole ? 0 : Float(pack.innerPixels[offset + pixel * 3 + channel]) / 255
                }
            }
        }
        let sound = try MLMultiArray(shape: [1, 40, 1024], dataType: .float32)
        let audioPointer = sound.dataPointer.bindMemory(to: Float.self, capacity: sound.count)
        for index in audio.indices { audioPointer[index] = audio[index] }
        let result = try renderer.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": image, "audio": sound]))
        guard let output = result.featureValue(for: rendererOutput)?.multiArrayValue,
              output.shape.map(\.intValue) == [1, 3, 288, 288] else { throw AvatarError.invalidPack("renderer output") }
        let values = try Self.floats(output), count = 288 * 288
        var bytes = [UInt8](repeating: 0, count: count * 3)
        for pixel in 0..<count {
            for channel in 0..<3 { bytes[pixel * 3 + channel] = UInt8(max(0, min(255, values[channel * count + pixel] * 255))) }
        }
        return Data(bytes)
    }
    private static func floats(_ array: MLMultiArray) throws -> [Float] {
        guard array.dataType == .float32 || array.dataType == .float16 else { throw AvatarError.invalidPack("model output dtype") }
        let shape = array.shape.map(\.intValue), strides = array.strides.map(\.intValue)
        let pointer = array.dataPointer.assumingMemoryBound(to: Float.self)
        let halfPointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        var expected = 1, contiguous = true
        for axis in shape.indices.reversed() {
            if shape[axis] > 1 && strides[axis] != expected { contiguous = false }
            expected *= shape[axis]
        }
        if contiguous {
            let result = array.dataType == .float32 ? Array(UnsafeBufferPointer(start: pointer, count: array.count)) : UnsafeBufferPointer(start: halfPointer, count: array.count).map(Float.init)
            guard result.allSatisfy(\.isFinite) else { throw AvatarError.invalidPack("nonfinite prediction") }
            return result
        }
        var result = [Float](repeating: 0, count: array.count)
        for linear in 0..<array.count {
            var remainder = linear, offset = 0
            for axis in shape.indices.reversed() { offset += (remainder % shape[axis]) * strides[axis]; remainder /= shape[axis] }
            result[linear] = array.dataType == .float32 ? pointer[offset] : Float(halfPointer[offset])
        }
        guard result.allSatisfy(\.isFinite) else { throw AvatarError.invalidPack("nonfinite prediction") }
        return result
    }
}
