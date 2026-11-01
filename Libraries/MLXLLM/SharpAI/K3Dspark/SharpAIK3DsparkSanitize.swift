// SharpAI fork overlay: k3_dspark checkpoint key sanitize. Not part of upstream mlx-swift-lm.
//
// Checkpoint keys have no "model." prefix (layers.N.*, context_proj.*, markov_head.*).
// The prefix is added here, as vLLM's hf_to_vllm_mapper does. embed_tokens and lm_head
// belong to the frozen target and are dropped. confidence_head is kept only when enabled.

import Foundation
import MLX

enum SharpAIK3DsparkSanitizer {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIK3DsparkConfiguration
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb") || key.contains("embed_tokens")
                || key.contains("lm_head")
            {
                continue
            }
            if key.contains("confidence_head") && !configuration.enableConfidenceHead {
                continue
            }
            let name = key.hasPrefix("model.") ? key : "model." + key
            out[name] = value
        }
        return out
    }
}
