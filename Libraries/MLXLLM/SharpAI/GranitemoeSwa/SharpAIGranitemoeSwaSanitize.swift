// SharpAI fork overlay: granitemoe_swa checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names except the routed experts. The fused `block_sparse_moe.input_linear`
// [experts, 2 * intermediate, hidden] is split into `switch_mlp.gate_proj` and `switch_mlp.up_proj`,
// and `output_linear` becomes `switch_mlp.down_proj`. The gate half comes first, as in vLLM.
// The router (`block_sparse_moe.router.layer`) and `shared_mlp` keep their HF names.

import Foundation
import MLX

public enum SharpAIGranitemoeSwaSanitizer {

    /// Drops RoPE buffers and the tied `lm_head`, and splits the fused expert tensors.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIGranitemoeSwaConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).block_sparse_moe"
            guard let inputWeight = sanitized.removeValue(forKey: "\(prefix).input_linear.weight")
            else { continue }

            let halves = inputWeight.split(parts: 2, axis: 1)
            sanitized["\(prefix).switch_mlp.gate_proj.weight"] = halves[0]
            sanitized["\(prefix).switch_mlp.up_proj.weight"] = halves[1]

            if let downWeight = sanitized.removeValue(forKey: "\(prefix).output_linear.weight") {
                sanitized["\(prefix).switch_mlp.down_proj.weight"] = downWeight
            }
        }

        return sanitized
    }
}
