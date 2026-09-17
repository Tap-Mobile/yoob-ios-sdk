//
//  Serve320Wav2Vec.swift
//  On-device FeatherHuBERT feature front-end for the Serve320 lane. The class
//  keeps its historical filename/type name to avoid a broad source refactor,
//  while FeatherHuBERT is the sole shipped runtime encoder.
//
//  Semantics (extract_l15.py:61-78):
//    window = [aligned_start - 8000, aligned_end + 8000), zero-padded
//    normalize per-utterance: (x - mean) / sqrt(var + 1e-7)  (HF Wav2Vec2FeatureExtractor)
//    hidden = CoreML FeatherHuBERT1024_fp16(win)
//    frame row: target sample = (2*gframe + 1 + shift) * 320, linear interp with
//    conv geometry (stride 320, first_center 199.5 — conv_geometry of config
//    kernels [10,3,3,3,3,2,2] / strides [5,2,2,2,2,2,2])
//  NOT included (next slice): replay._silence_feats 25-frame tail pad/trim.
//

import Foundation
import CoreML
import Accelerate

enum Serve320Wav2VecError: Error {
    case badInput(String)
}

final class Serve320Wav2Vec {
    static let registryID = "serve320.featherhubert"
    static let displayName = "Feather 6.5MB"
    static let sr = 16000
    static let hop = 320
    static let context = 8000
    /// HuBERT-compatible convolution geometry (stride 320, receptive field 400).
    static let convStride: Double = 320
    static let convFirstCenter: Double = 199.5

    let model: MLModel

    private init(model: MLModel) {
        self.model = model
    }

    private static func configuredComputeUnits() -> MLComputeUnits {
        switch ProcessInfo.processInfo.environment[
            "AVATAR_FEATHER_COMPUTE"
        ]?.lowercased() {
        case "cpu": return .cpuOnly
        case "ane": return .cpuAndNeuralEngine
        case "gpu": return .cpuAndGPU
        default: return .all
        }
    }

    static func load() async throws -> Serve320Wav2Vec {
        // Keep `.all` as the validated physical-phone default. Explicit
        // AVATAR_FEATHER_COMPUTE overrides remain available for placement
        // measurement without adding another user-visible model choice.
        let model = try await ModelRegistry.shared.load(
            registryID, computeUnits: configuredComputeUnits())
        return Serve320Wav2Vec(model: model)
    }

    /// extract_l15.py extract(): pcm 16kHz mono f32 -> (nFrames,1024) fp16 (flat).
    /// aligned_start/end in samples; gframe = global_frame_start + i.
    func extract(pcm: [Float], nFrames: Int, shift: Int,
                 alignedStart: Int = 0, alignedEnd: Int? = nil,
                 globalFrameStart: Int = 0) throws -> [Float16] {
        let aEnd = alignedEnd ?? (alignedStart + nFrames * 640)
        let ws = alignedStart - Self.context
        let we = aEnd + Self.context
        // extract_l15.py:67-71 — zero-pad window, copy the overlapping audio span
        var win = [Float](repeating: 0, count: we - ws)
        let lo = max(ws, 0), hi = min(we, pcm.count)
        if hi > lo {
            win.withUnsafeMutableBufferPointer { dst in
                pcm.withUnsafeBufferPointer { src in
                    dst.baseAddress!.advanced(by: lo - ws)
                        .update(from: src.baseAddress!.advanced(by: lo), count: hi - lo)
                }
            }
        }
        // Training-time waveform normalization: (x - mean) / sqrt(var + 1e-7)
        var mean: Float = 0
        vDSP_meanv(win, 1, &mean, vDSP_Length(win.count))
        var centered = [Float](repeating: 0, count: win.count)
        var negMean = -mean
        vDSP_vsadd(win, 1, &negMean, &centered, 1, vDSP_Length(win.count))
        var varPop: Float = 0
        vDSP_measqv(centered, 1, &varPop, vDSP_Length(win.count))   // population variance
        var invStd = 1 / sqrt(varPop + 1e-7)
        var norm = [Float](repeating: 0, count: win.count)
        vDSP_vsmul(centered, 1, &invStd, &norm, 1, vDSP_Length(win.count))

        let input = try MLMultiArray(shape: [1, NSNumber(value: win.count)], dataType: .float32)
        let ptr = input.dataPointer.bindMemory(to: Float.self, capacity: win.count)
        norm.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: win.count)
        }
        let out = try predict(input)
        guard let hidden = out.featureValue(for: "hidden")?.multiArrayValue else {
            throw Serve320Wav2VecError.badInput("no hidden output")
        }
        let t = hidden.shape[1].intValue
        guard hidden.shape.count == 3, hidden.shape[2].intValue == 1024 else {
            throw Serve320Wav2VecError.badInput("hidden shape \(hidden.shape)")
        }
        let hPtr = hidden.dataPointer.bindMemory(to: Float.self, capacity: t * 1024)

        // extract_l15.py:76-78 + _interp (51-58): target sample = (2*g+1+shift)*hop,
        // pos = (target - ws - first_center) / stride, fp64 position math.
        var rows = [Float16](repeating: 0, count: nFrames * 1024)
        for i in 0..<nFrames {
            let g = globalFrameStart + i
            let target = Double((2 * g + 1 + shift) * Self.hop)
            let pos = (target - Double(ws) - Self.convFirstCenter) / Self.convStride
            guard pos >= 0, pos <= Double(t - 1) else {
                throw Serve320Wav2VecError.badInput("frame \(i) target outside encoder support")
            }
            let left = Int(pos.rounded(.down))
            let right = min(left + 1, t - 1)
            let w = Float(pos - Double(left))
            let lBase = left * 1024
            let rBase = right * 1024
            let oBase = i * 1024
            for c in 0..<1024 {
                let v = hPtr[lBase + c] * (1 - w) + hPtr[rBase + c] * w
                rows[oBase + c] = Float16(v)
            }
        }
        return rows
    }

    /// Sync prediction — in a non-async context only the sync overload of
    /// MLModel.prediction(from:) is visible (the async one would shadow it here).
    private func predict(_ input: MLMultiArray) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio": MLFeatureValue(multiArray: input),
        ])
        return try model.prediction(from: provider)
    }

}
