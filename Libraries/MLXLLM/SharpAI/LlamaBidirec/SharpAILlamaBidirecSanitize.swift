// SharpAI fork overlay: llama_bidirec checkpoint key sanitization. Not part of upstream
// mlx-swift-lm.
//
// The module tree is model.embed_tokens, model.layers.N, model.norm. Checkpoints may omit the
// leading "model." (the bare base-model layout). Those keys get the prefix added here.
// Keys that this model has no module for are dropped: rotary buffers, lm_head, and score.
//
// UNVERIFIED: whether the target checkpoint uses the bare or the "model." layout is not known
// from this repo. Both are handled.

import Foundation
import MLX

enum SharpAILlamaBidirecSanitizer {

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
        if key.hasPrefix("embed_tokens.") || key.hasPrefix("layers.") || key.hasPrefix("norm.") {
            return "model." + key
        }
        return key
    }

    static func isDroppedKey(_ key: String) -> Bool {
        if key.contains("rotary_emb.inv_freq") {
            return true
        }
        return key.hasPrefix("lm_head.") || key.hasPrefix("score.")
            || key.hasPrefix("model.score.")
    }
}
