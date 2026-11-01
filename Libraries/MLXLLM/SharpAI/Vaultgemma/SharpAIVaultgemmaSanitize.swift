//
//  SharpAIVaultgemmaSanitize.swift
//  MLXLLM
//
//  Checkpoint key cleanup for `vaultgemma`. Keys keep their HF names. The HF model deletes
//  the post-attention and post-feedforward norms, so any copies in a checkpoint are dropped.
//

import Foundation
import MLX

enum SharpAIVaultgemmaSanitizer {

    /// Drops rotary buffers, post-block norms, and the tied `lm_head`.
    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIVaultgemmaConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        sanitized = sanitized.filter { key, _ in
            !key.contains(".post_attention_layernorm.")
                && !key.contains(".post_feedforward_layernorm.")
        }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { key, _ in
                !key.split(separator: ".").contains("lm_head")
            }
        }

        return sanitized
    }
}
