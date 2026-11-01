// SharpAI fork overlay: cohere_asr checkpoint key and layout sanitization. Not part of
// upstream mlx-swift-lm.
//
// Keys keep their HF names, except the subsampling conv list. HF interleaves relu modules
// there, so the Swift list is shorter. Conv weights change from PyTorch layout to MLX layout.
// UNVERIFIED: assumes an unconverted PyTorch checkpoint. An MLX-converted one would be
// transposed again. Check against a real checkpoint before relying on this.

import Foundation
import MLX

public enum SharpAICohereAsrSanitizer {

    /// HF `model.encoder.subsampling.layers.<i>` index to Swift index. Relu layers are gone.
    static let subsamplingLayerIndex: [Int: Int] = [0: 0, 2: 1, 3: 2, 5: 3, 6: 4]

    static let subsamplingPrefix = "model.encoder.subsampling.layers."

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAICohereAsrConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if shouldDrop(key) {
                continue
            }
            let renamed = remapKey(key)
            sanitized[renamed] = transposeConvWeight(renamed, value)
        }

        if configuration.tieWordEmbeddings && sanitized["proj_out.weight"] == nil {
            guard let embedding = sanitized["model.decoder.embed_tokens.weight"] else {
                throw SharpAICohereAsrError.missingWeight("model.decoder.embed_tokens.weight")
            }
            sanitized["proj_out.weight"] = embedding
        }

        return sanitized
    }

    /// Drops buffers and non-model tensors: BatchNorm counters, the preprocessor featurizer, and
    /// the relative-position inverse frequencies. The Swift model computes those inline.
    static func shouldDrop(_ key: String) -> Bool {
        key.hasSuffix("num_batches_tracked")
            || key.hasPrefix("preprocessor.")
            || key.contains("encode_positions.inv_freq")
            || key.contains("rotary_emb.inv_freq")
    }

    static func remapKey(_ key: String) -> String {
        guard key.hasPrefix(subsamplingPrefix) else {
            return key
        }
        let rest = key.dropFirst(subsamplingPrefix.count)
        guard let dot = rest.firstIndex(of: "."),
            let index = Int(rest[..<dot]),
            let mapped = subsamplingLayerIndex[index]
        else {
            return key
        }
        return subsamplingPrefix + String(mapped) + String(rest[dot...])
    }

    /// PyTorch conv weights are (out, in / groups, k...). MLX wants (out, k..., in / groups).
    static func transposeConvWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        guard key.hasSuffix(".weight") else {
            return value
        }
        switch value.ndim {
        case 4:
            return value.transposed(0, 2, 3, 1)
        case 3:
            return value.transposed(0, 2, 1)
        default:
            return value
        }
    }
}
