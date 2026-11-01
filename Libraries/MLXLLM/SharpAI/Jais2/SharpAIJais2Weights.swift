//
//  SharpAIJais2Weights.swift
//  MLXLLM
//
//  Key cleanup for `jais2` checkpoints. Weight keys keep the Hugging Face names.
//  UNVERIFIED: the rules below follow the LLaMA convention used by LlamaModel.
//  Check them against a real checkpoint index.
//

import Foundation
import MLX

enum SharpAIJais2Weights {

    /// Drops precomputed rotary tables and, for tied embeddings, the separate `lm_head`.
    static func sanitize(
        weights: [String: MLXArray], tieWordEmbeddings: Bool
    ) -> [String: MLXArray] {
        weights.filter { key, _ in
            if key.contains("self_attn.rotary_emb.inv_freq") {
                return false
            }
            if tieWordEmbeddings && key.hasPrefix("lm_head.") {
                return false
            }
            return true
        }
    }
}
