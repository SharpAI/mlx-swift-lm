// SharpAI fork overlay: vibevoice_asr checkpoint key and layout sanitization. Not part of
// upstream mlx-swift-lm.
//
// Keys keep their HF names. The only rename strips a leading "model." when the rest starts with a
// known top-level module. Tokenizer conv weights change from PyTorch (out, in / groups, k) to MLX
// (out, k, in / groups).
// UNVERIFIED: the key names are taken from transformers' module tree, not from a real checkpoint.
// UNVERIFIED: this assumes weight_norm was already folded into plain weights. The reference
// modeling code has no weight_norm, so a checkpoint that still has original0/original1 or
// weight_g/weight_v will not load.

import Foundation
import MLX
import MLXLMCommon

public enum SharpAIVibevoiceAsrSanitizer {

    static let topLevelPrefixes = [
        "language_model.",
        "multi_modal_projector.",
        "acoustic_tokenizer_encoder.",
        "semantic_tokenizer_encoder.",
    ]

    static let tokenizerPrefixes = [
        "acoustic_tokenizer_encoder.",
        "semantic_tokenizer_encoder.",
    ]

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIVibevoiceAsrConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            let renamed = remapKey(key)
            if shouldDrop(renamed, configuration: configuration) {
                continue
            }
            sanitized[renamed] = transposeConvWeight(renamed, value)
        }

        return sanitized
    }

    /// Drops rotary buffers, tokenizer decoder weights the ASR path never uses, and the output
    /// head when embeddings are tied.
    static func shouldDrop(_ key: String, configuration: SharpAIVibevoiceAsrConfiguration) -> Bool {
        if key.contains("rotary_emb.inv_freq") {
            return true
        }
        // UNVERIFIED: the ASR checkpoint may not contain these. They are dropped if it does.
        if key.hasPrefix("acoustic_tokenizer_decoder.") || key.hasPrefix("semantic_tokenizer_decoder.") {
            return true
        }
        if configuration.text.tieWordEmbeddings && key == "language_model.lm_head.weight" {
            return true
        }
        return false
    }

    static func remapKey(_ key: String) -> String {
        let prefix = "model."
        guard key.hasPrefix(prefix) else {
            return key
        }
        let rest = String(key.dropFirst(prefix.count))
        return topLevelPrefixes.contains(where: { rest.hasPrefix($0) }) ? rest : key
    }

    /// PyTorch Conv1d weights are (out, in / groups, k). MLX wants (out, k, in / groups).
    static func transposeConvWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        guard key.hasSuffix(".weight"), value.ndim == 3 else {
            return value
        }
        guard tokenizerPrefixes.contains(where: { key.hasPrefix($0) }) else {
            return value
        }
        return value.transposed(0, 2, 1)
    }
}
