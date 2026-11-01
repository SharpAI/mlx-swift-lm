//
//  SharpAISarvamMlaSanitize.swift
//  MLXLLM
//
//  Checkpoint key cleanup for `sarvam_mla`. Keeps HF names and only stacks
//  per-expert weights into the `switch_mlp` layout.
//  UNVERIFIED: the per-expert key names (`mlp.experts.N.<proj>`) follow the
//  DeepSeek-V3 convention. No Sarvam checkpoint index was read.
//

import Foundation
import MLX

enum SharpAISarvamMlaSanitizer {
    static let expertProjections = ["gate_proj", "down_proj", "up_proj"]
    static let expertSuffixes = ["weight", "scales", "biases"]

    static func sanitize(
        weights: [String: MLXArray], configuration: SharpAISarvamMlaConfiguration
    ) -> [String: MLXArray] {
        var result = weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }

        guard configuration.numExperts > 0 else {
            return result
        }

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for proj in expertProjections {
                for suffix in expertSuffixes {
                    let firstKey = "\(prefix).mlp.experts.0.\(proj).\(suffix)"
                    guard result[firstKey] != nil else { continue }

                    var parts: [MLXArray] = []
                    for expert in 0 ..< configuration.numExperts {
                        let key = "\(prefix).mlp.experts.\(expert).\(proj).\(suffix)"
                        guard let part = result.removeValue(forKey: key) else { continue }
                        parts.append(part)
                    }
                    result["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = stacked(parts)
                }
            }
        }

        return result
    }
}
