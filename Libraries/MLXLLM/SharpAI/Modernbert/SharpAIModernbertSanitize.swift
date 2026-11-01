// SharpAI fork overlay: modernbert checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// The module tree is model.embeddings, model.layers.N, model.final_norm. Bare ModernBertModel
// checkpoints omit the leading "model.". Those keys get the prefix added here.
// Keys with no module here are dropped: the masked-LM head (head.*, decoder.*) and rotary or
// position buffers.
//
// UNVERIFIED: the ColBERT projection key is not known from this repo. Keys under "linear." are
// dropped on that guess, so the projection is not loaded.

import Foundation
import MLX

enum SharpAIModernbertSanitizer {

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            guard !isDroppedKey(key) else { continue }
            sanitized[mappedKey(key)] = value
        }
        return sanitized
    }

    static func mappedKey(_ key: String) -> String {
        if key.hasPrefix("model.") {
            return key
        }
        if key.hasPrefix("embeddings.") || key.hasPrefix("layers.")
            || key.hasPrefix("final_norm.")
        {
            return "model." + key
        }
        return key
    }

    static func isDroppedKey(_ key: String) -> Bool {
        if key.contains("rotary_emb") || key.contains("position_ids") {
            return true
        }
        let headPrefixes = ["head.", "decoder.", "model.head.", "model.decoder.", "linear."]
        return headPrefixes.contains { key.hasPrefix($0) }
    }
}
