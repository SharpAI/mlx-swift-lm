// SharpAI fork overlay: step3p7 checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Text backbone rules mirror the step3p5 sanitizer in mlx-lm (Model.sanitize), since text_config is step3p5.
// Module names match SharpAIStep3p7TextModel in Libraries/MLXLLM/SharpAI/Step3p7.
// UNVERIFIED: no step3p7 checkpoint index is on disk, so the key names are assumed to match step3p5.
// Vision keys pass through unchanged. lm_head and tied-embedding filtering are not applied.
// Vanilla checkpoints store norm weights zero-centered, so 1 is added to them, as mlx-lm does.

import Foundation
import MLX

public enum SharpAIStep3p7Sanitizer {

    private static let projections = ["gate_proj", "up_proj", "down_proj"]
    private static let suffixes = ["weight", "scales", "biases"]

    /// Raw source name and its MLX target name, from mlx-lm `Model.sanitize`.
    /// A key that has the source name but not the target name marks a vanilla checkpoint.
    private static let vanillaMarkers: [(source: String, target: String)] = [
        (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
        (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
        (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
        (".moe.gate.", ".mlp.gate.gate."),
        (".moe.router_bias", ".mlp.gate.router_bias"),
        (".share_expert.", ".mlp.share_expert."),
    ]

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIStep3p7Configuration
    ) -> [String: MLXArray] {
        let textConfig = configuration.textConfig
        let hiddenLayers = textConfig.hiddenLayers

        let isVanilla = weights.keys.contains { key in
            vanillaMarkers.contains { key.contains($0.source) && !key.contains($0.target) }
        }

        // Drop rotary buffers, expert_bias, and MTP layers (index >= num_hidden_layers).
        // Rename share_expert into moe.share_expert.
        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.contains("expert_bias") {
                continue
            }
            if let layer = layerIndex(of: key), layer >= hiddenLayers {
                continue
            }
            let newKey = key.replacingOccurrences(
                of: ".share_expert.", with: ".moe.share_expert.")
            renamed[newKey] = value
        }

        if isVanilla {
            for key in Array(renamed.keys) where key.hasSuffix(".weight") && key.contains("norm") {
                if let value = renamed[key] {
                    renamed[key] = value + Float(1)
                }
            }
        }

        guard let numExperts = textConfig.moeNumExperts else { return renamed }

        for layer in 0 ..< hiddenLayers {
            let prefix = "model.layers.\(layer).moe"

            // Packed format: already stacked with a leading expert axis.
            for proj in projections {
                for suffix in suffixes {
                    if let value = renamed.removeValue(forKey: "\(prefix).\(proj).\(suffix)") {
                        renamed["\(prefix).switch_mlp.\(proj).\(suffix)"] = value
                    }
                }
            }

            // Per-expert format: stack experts into switch_mlp. Layers without experts are skipped.
            for proj in projections {
                for suffix in suffixes {
                    guard renamed["\(prefix).experts.0.\(proj).\(suffix)"] != nil else { continue }
                    let toJoin = (0 ..< numExperts).compactMap { expert in
                        renamed.removeValue(forKey: "\(prefix).experts.\(expert).\(proj).\(suffix)")
                    }
                    renamed["\(prefix).switch_mlp.\(proj).\(suffix)"] = MLX.stacked(toJoin)
                }
            }
        }

        return renamed
    }

    /// Layer index for `model.layers.<n>.*` keys, or nil for other keys.
    private static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard parts.count > 2, parts[0] == "model", parts[1] == "layers" else { return nil }
        return Int(parts[2])
    }
}
