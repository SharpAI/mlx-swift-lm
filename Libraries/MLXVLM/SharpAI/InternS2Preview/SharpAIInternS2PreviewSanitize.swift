// SharpAI fork overlay: intern_s2_preview key helpers. Not part of upstream mlx-swift-lm.
//
// vLLM maps "model.time_series." and "time_series." to None, so those tensors are dropped.
// Expert fusion matches Qwen35MoE. Other keys keep their HF names.

import Foundation
import MLX

public enum SharpAIInternS2PreviewSanitizer {

    /// Checkpoint prefixes for the time-series tower. The text model has no such module.
    /// The `language_model.model.` form is what Qwen35's prepareTarget makes of `model.time_series.`.
    public static let excludedPrefixes = [
        "model.time_series.", "time_series.", "language_model.model.time_series.",
    ]

    public static func dropTimeSeries(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { entry in
            !excludedPrefixes.contains { entry.key.hasPrefix($0) }
        }
    }

    /// Splits `experts.gate_up_proj` into `switch_mlp` gate and up weights, as Qwen35MoE does.
    public static func fuseExperts(
        _ weights: [String: MLXArray], hiddenLayers: Int
    ) -> [String: MLXArray] {
        var remapped = weights
        for layer in 0 ..< hiddenLayers {
            let prefixes = [
                "model.language_model.layers.\(layer).mlp",
                "language_model.model.layers.\(layer).mlp",
            ]
            for prefix in prefixes {
                let gateUpKey = "\(prefix).experts.gate_up_proj"
                if let gateUp = remapped.removeValue(forKey: gateUpKey) {
                    let mid = gateUp.dim(-2) / 2
                    remapped["\(prefix).switch_mlp.gate_proj.weight"] =
                        gateUp[.ellipsis, ..<mid, 0...]
                    remapped["\(prefix).switch_mlp.up_proj.weight"] =
                        gateUp[.ellipsis, mid..., 0...]

                    let downProjKey = "\(prefix).experts.down_proj"
                    if let downProj = remapped.removeValue(forKey: downProjKey) {
                        remapped["\(prefix).switch_mlp.down_proj.weight"] = downProj
                    }
                }
            }
        }
        return remapped
    }
}
