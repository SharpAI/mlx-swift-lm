// SharpAI fork overlay: muse_spark_dspark checkpoint key sanitize. Not part of upstream mlx-swift-lm.
//
// Mirrors vLLM's MuseSparkDSparkForCausalLM.load_weights. Checkpoint keys have no "model."
// prefix, so it is added here, except for lm_head. d2t becomes draft_id_to_target_id, and t2d is
// dropped (training only). embed_tokens and mask_embedding belong to the target and are dropped.
// The confidence head and residual gates are dropped when the config disables them.

import Foundation
import MLX

enum SharpAIMuseSparkDsparkSanitizer {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIMuseSparkDsparkConfiguration
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb") || key.contains("embed_tokens")
                || key.contains("mask_embedding") || key.contains("t2d")
            {
                continue
            }
            if key.contains("d2t") {
                out["draft_id_to_target_id"] = value
                continue
            }
            if key.contains("confidence_head") && !configuration.enableConfidenceHead {
                continue
            }
            if (key.contains("add_attn") || key.contains("add_ffn"))
                && !configuration.useResidualGate
            {
                continue
            }
            if key.contains("lm_head") {
                out[key] = value
                continue
            }
            let name = key.hasPrefix("model.") ? key : "model." + key
            out[name] = value
        }
        return out
    }
}
