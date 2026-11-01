// SharpAI fork overlay: cohere_asr checkpoint key and layout sanitization. Not part of
// upstream mlx-swift-lm.
//
// Keys keep their HF names, except the subsampling conv list. HF interleaves relu modules
// there, so the Swift list is shorter. Conv weights change from PyTorch layout to MLX layout.
// The layout is detected once from the first subsampling conv: (out, k, k, 1) means the
// checkpoint is already in MLX layout and no conv weight is transposed. Anything else is
// treated as PyTorch. Check against a real checkpoint before relying on this.

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
        let alreadyMLXLayout = isMLXLayout(weights, configuration: configuration)

        for (key, value) in weights {
            if shouldDrop(key) {
                continue
            }
            let renamed = remapKey(key)
            sanitized[renamed] = alreadyMLXLayout ? value : transposeConvWeight(renamed, value)
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

    /// True when the first subsampling conv is (out, k, k, 1). PyTorch stores it (out, 1, k, k).
    /// A kernel size of 1 is ambiguous and counts as PyTorch.
    static func isMLXLayout(
        _ weights: [String: MLXArray], configuration: SharpAICohereAsrConfiguration
    ) -> Bool {
        let kernel = configuration.encoderConfiguration.subsamplingConvKernelSize
        guard kernel > 1, let first = weights[subsamplingPrefix + "0.weight"], first.ndim == 4
        else {
            return false
        }
        return first.dim(1) == kernel && first.dim(2) == kernel && first.dim(3) == 1
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
