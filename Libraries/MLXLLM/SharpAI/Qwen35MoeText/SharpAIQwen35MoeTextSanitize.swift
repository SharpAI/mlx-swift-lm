// SharpAI fork overlay: qwen3_5_moe_text expert stacking. Not part of upstream mlx-swift-lm.
//
// The upstream text model (`Qwen35TextModel`) remaps keys but does not stack experts. The MoE
// wrapper (`Qwen35MoEModel`) does, for the wrapped `language_model.` layout (main layers in
// Step 3, MTP heads in Step 4). This mirrors both steps for the text layout, where the
// `language_model.` prefix is absent: `model.layers.N.mlp` and `mtp.{N}.layers.0.mlp` etc.
// UNVERIFIED: no official text-only qwen3_5_moe_text checkpoint is on this machine. The key
// names follow the wrapped Qwen3.5 MoE checkpoints with the prefix removed. The MTP prefixes
// were checked against the upstream wrapper's Step 4 only, not against a text-only MTP checkpoint.

import Foundation
import MLX
import MLXLMCommon

public enum SharpAIQwen35MoeTextExperts {

    static let projections = ["gate_proj", "up_proj", "down_proj"]
    static let parameters = ["weight", "scales", "biases", "weight_scale_inv"]

    /// Stacks per-expert tensors into `switch_mlp` tensors for the main decoder layers and,
    /// when `mtpLayers` is positive, for the MTP heads. Keys that are already stacked, or that
    /// have no MoE block, pass through unchanged.
    public static func stack(
        _ weights: [String: MLXArray], layers: Int, numExperts: Int, mtpLayers: Int = 0
    ) -> [String: MLXArray] {
        guard numExperts > 0 else { return weights }
        let streaming = ExpertStreamingConfig.shared.isEnabled
        var out = weights
        for layer in 0 ..< layers {
            let prefix = "model.layers.\(layer).mlp"
            stackFusedGateUp(&out, prefix: prefix)
            stackPerExpert(&out, prefix: prefix, numExperts: numExperts, streaming: streaming)
        }
        for layer in 0 ..< mtpLayers {
            for prefix in mtpPrefixes(layer) {
                stackFusedGateUp(&out, prefix: prefix)
                stackPerExpert(&out, prefix: prefix, numExperts: numExperts, streaming: streaming)
            }
        }
        return out
    }

    /// MTP MoE prefixes, mirroring `Qwen35MoEModel` Step 4 with the `language_model.` segment
    /// removed. The duplicate for layer 0 is dropped, so no prefix is processed twice.
    static func mtpPrefixes(_ layer: Int) -> [String] {
        var seen = Set<String>()
        return [
            "mtp.\(layer).layers.0.mlp",
            "mtp.layers.0.mlp",
            "mtp.layers.\(layer).mlp",
        ].filter { seen.insert($0).inserted }
    }

    /// Format A: a pre-stacked `experts.gate_up_proj` is split into gate and up projections.
    private static func stackFusedGateUp(_ weights: inout [String: MLXArray], prefix: String) {
        guard let gateUp = weights.removeValue(forKey: "\(prefix).experts.gate_up_proj") else {
            return
        }
        let mid = gateUp.dim(-2) / 2
        weights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[.ellipsis, ..<mid, 0...]
        weights["\(prefix).switch_mlp.up_proj.weight"] = gateUp[.ellipsis, mid..., 0...]
        if let down = weights.removeValue(forKey: "\(prefix).experts.down_proj") {
            weights["\(prefix).switch_mlp.down_proj.weight"] = down
        }
    }

    /// Format B: per-expert `experts.N.{proj}.{param}` tensors are stacked along a new expert axis.
    /// Under expert streaming the weight is not stacked; the per-expert keys are still dropped.
    private static func stackPerExpert(
        _ weights: inout [String: MLXArray], prefix: String, numExperts: Int, streaming: Bool
    ) {
        for projection in projections {
            for parameter in parameters {
                let names = (0 ..< numExperts).map {
                    "\(prefix).experts.\($0).\(projection).\(parameter)"
                }
                let arrays = names.compactMap { weights[$0] }
                guard arrays.count == numExperts else { continue }
                if !(streaming && parameter == "weight") {
                    weights["\(prefix).switch_mlp.\(projection).\(parameter)"] =
                        MLX.stacked(arrays)
                }
                for name in names {
                    weights.removeValue(forKey: name)
                }
            }
        }
    }
}
