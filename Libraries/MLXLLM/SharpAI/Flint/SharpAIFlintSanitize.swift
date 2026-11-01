// SharpAI fork overlay: flint checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Per-expert tensors `mlp.experts.<e>.<proj>` are stacked into `mlp.switch_mlp.<proj>`.
// Other keys keep their HF names, including `self_attn.softmax_offset` and `mlp.gate.weight`.
//
// UNVERIFIED: no safetensors index was available. Key names come from the vLLM GGUF name map
// and the vLLM model's parameter names, not from a real checkpoint.

import Foundation
import MLX

public enum SharpAIFlintSanitizer {

    /// Drops rotary buffers and stacks per-expert weights. A layer whose expert set is incomplete
    /// is left unchanged, so the loader reports the missing keys.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIFlintConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(kind)"
                    guard sanitized[firstKey] != nil else { continue }

                    let keys = (0 ..< configuration.numExperts).map {
                        "\(prefix).mlp.experts.\($0).\(proj).\(kind)"
                    }
                    guard keys.allSatisfy({ sanitized[$0] != nil }) else { continue }

                    let toJoin = keys.map { sanitized.removeValue(forKey: $0)! }
                    sanitized["\(prefix).mlp.switch_mlp.\(proj).\(kind)"] = MLX.stacked(toJoin)
                }
            }
        }

        return sanitized
    }
}
