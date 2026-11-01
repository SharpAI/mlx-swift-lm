//
//  SharpAIIquestloopcoderWeights.swift
//  MLXLLM
//
//  Key cleanup for `iquestloopcoder` checkpoints. Weight keys keep the Hugging Face names,
//  which match the IQuestLab reference: `model.gate_projections.{i}.{weight,bias}` for the loop
//  gates, plus the usual Llama-style `self_attn.{q,k,v,o}_proj` and `mlp.{gate,up,down}_proj`.
//  No renaming is needed; the gate tensors load directly.
//

import Foundation
import MLX

enum SharpAIIquestloopcoderWeights {

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
