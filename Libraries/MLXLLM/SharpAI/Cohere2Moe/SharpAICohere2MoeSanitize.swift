// SharpAI fork overlay: cohere2_moe checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the per-expert key layout (`mlp.experts.<e>.<proj>`) follows the GLM4-MoE
// convention in this repo. Shared-expert keys are not remapped. Router keys keep their HF names.
// transformers v5.10.2 keeps experts fused in memory (`mlp.experts.gate_up_proj`,
// `mlp.experts.down_proj`, no `.weight`). That layout is not mapped here, so it fails with
// unexpectedExpertLayout rather than loading unconverted experts. Confirm the on-disk layout
// against a real cohere2_moe checkpoint index.

import Foundation
import MLX

public enum SharpAICohere2MoeSanitizer {

    /// Removes non-weight buffers and stacks per-expert weights into `switch_mlp`.
    /// Other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAICohere2MoeConfiguration
    ) throws -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        guard let numExperts = configuration.numExperts, numExperts > 0 else {
            return sanitized
        }

        var stackedCount = 0
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(kind)"
                    guard sanitized[firstKey] != nil else { continue }

                    let toJoin = try (0 ..< numExperts).map { expert -> MLXArray in
                        let key = "\(prefix).mlp.experts.\(expert).\(proj).\(kind)"
                        guard let weight = sanitized.removeValue(forKey: key) else {
                            throw SharpAICohere2MoeError.missingExpertWeight(key)
                        }
                        return weight
                    }
                    sanitized["\(prefix).mlp.switch_mlp.\(proj).\(kind)"] = MLX.stacked(toJoin)
                    stackedCount += 1
                }
            }
        }

        // Fail on an unknown layout instead of loading unconverted experts.
        let hasSwitchMLP = sanitized.keys.contains { $0.contains(".mlp.switch_mlp.") }
        if stackedCount == 0 && !hasSwitchMLP {
            throw SharpAICohere2MoeError.unexpectedExpertLayout(
                "no mlp.experts.<e> or mlp.switch_mlp weights for \(numExperts) experts")
        }
        if let leftover = sanitized.keys.sorted().first(where: { $0.contains(".mlp.experts.") }) {
            throw SharpAICohere2MoeError.unexpectedExpertLayout(leftover)
        }

        return sanitized
    }
}
