//
//  SharpAIStep3p5Sanitize.swift
//  MLXLLM
//
//  Checkpoint key sanitization for step3p5. Not part of upstream mlx-swift-lm.
//
//  - Drops rotary buffers, vLLM-ignored `expert_bias` keys, and MTP layers (index >= num_hidden_layers).
//  - Renames `<layer>.share_expert.*` to `<layer>.moe.share_expert.*`, as vLLM hf_to_vllm_mapper does.
//  - Stacks routed experts into `moe.switch_mlp.*`. Accepts the packed format
//    (`moe.{gate,up,down}_proj.weight`, leading expert axis) and the per-expert format
//    (`moe.experts.{e}.{gate,up,down}_proj.weight`).
//  UNVERIFIED: no checkpoint index was read. Expert key formats come from vLLM step3p5.py.
//

import Foundation
import MLX
import MLXLMCommon

enum SharpAIStep3p5Sanitizer {

    private static let projections = ["gate_proj", "up_proj", "down_proj"]
    private static let suffixes = ["weight", "scales", "biases"]

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIStep3p5Configuration
    ) -> [String: MLXArray] {
        var out = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        out = out.filter { entry in
            let key = entry.key
            if key.contains("rotary_emb.inv_freq") || key.contains("expert_bias") {
                return false
            }
            if let layer = layerIndex(of: key) {
                return layer < configuration.hiddenLayers
            }
            return true
        }

        var renamed: [String: MLXArray] = [:]
        for (key, value) in out {
            // Skip keys already renamed, so a second pass does not produce moe.moe.
            let newKey =
                key.contains(".moe.share_expert.")
                ? key
                : key.replacingOccurrences(of: ".share_expert.", with: ".moe.share_expert.")
            renamed[newKey] = value
        }

        let numExperts = configuration.moeNumExperts
        for layer in 0 ..< configuration.hiddenLayers where configuration.isMoELayer(layer) {
            let prefix = "model.layers.\(layer).moe"

            // Packed format: already stacked with a leading expert axis.
            for proj in projections {
                for suffix in suffixes {
                    if let value = renamed.removeValue(forKey: "\(prefix).\(proj).\(suffix)") {
                        renamed["\(prefix).switch_mlp.\(proj).\(suffix)"] = value
                    }
                }
            }

            // Per-expert format: missing experts are skipped here. A short checkpoint fails at load.
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
