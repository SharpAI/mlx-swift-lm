// SharpAI fork overlay: voxtral_realtime checkpoint key and layout sanitization. Not part of
// upstream mlx-swift-lm.
//
// Keys keep their HF names after one optional leading `model.` is removed. The output head is
// tied to the embedding, so any `lm_head.weight` is dropped. Conv1d weights change from PyTorch
// (out, in, k) to MLX (out, k, in).
// UNVERIFIED: assumes an unconverted PyTorch checkpoint with HF key names. Check against a real
// checkpoint before relying on this.

import Foundation
import MLX

public enum SharpAIVoxtralRealtimeSanitizer {

    static let embeddingKey = "language_model.model.embed_tokens.weight"

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIVoxtralRealtimeConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            let renamed = remapKey(key)
            if shouldDrop(renamed) {
                continue
            }
            sanitized[renamed] = transposeConvWeight(renamed, value)
        }

        guard sanitized[embeddingKey] != nil else {
            throw SharpAIVoxtralRealtimeError.missingWeight(embeddingKey)
        }
        return sanitized
    }

    /// Drops the tied head and buffers the Swift model does not hold.
    static func shouldDrop(_ key: String) -> Bool {
        key == "lm_head.weight"
            || key.hasSuffix("num_batches_tracked")
            || key.contains("rotary_emb.inv_freq")
    }

    static func remapKey(_ key: String) -> String {
        let prefix = "model."
        return key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : key
    }

    /// PyTorch Conv1d weights are (out, in, k). MLX wants (out, k, in).
    static func transposeConvWeight(_ key: String, _ value: MLXArray) -> MLXArray {
        guard key.hasPrefix("audio_tower.conv"), key.hasSuffix(".weight"), value.ndim == 3 else {
            return value
        }
        return value.transposed(0, 2, 1)
    }
}
