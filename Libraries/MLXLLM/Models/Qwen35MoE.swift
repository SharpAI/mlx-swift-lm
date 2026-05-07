//
//  Qwen35MoE.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/9.
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5_moe.py
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Qwen35Configuration: Codable, Sendable {
    var modelType: String
    var textConfig: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decode(String.self, forKey: .modelType)

        if let textConfig = try container.decodeIfPresent(
            Qwen35TextConfiguration.self, forKey: .textConfig)
        {
            self.textConfig = textConfig
        } else {
            self.textConfig = try Qwen35TextConfiguration(from: decoder)
        }
    }
}

public class Qwen35MoEModel: Qwen35Model {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // ── Step 1: FP8 dequantization (official Qwen3.6-35B-A3B-FP8 checkpoint) ──
        // The FP8 release stores quantized weights alongside weight_scale_inv tensors.
        // We preserve them and stack them so they can be lazily dequantized in SwitchLinear.
        // ── Step 2: Key remapping ──
        var newWeights = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") {
                continue
            }
            var key = key
            if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if !key.hasPrefix("language_model.") {
                key = "language_model." + key
            }
            newWeights[key] = value
        }

        // ── Step 3: MoE expert weight stacking (main layers) ──
        // Format A: community 4-bit checkpoints ship a pre-stacked "gate_up_proj" → split into gate/up
        // Format B: FP8/BF16 official checkpoints ship per-expert "experts.N.{gate,up,down}_proj" → stack
        let nExperts = languageModel.configuration.numExperts
        for l in 0 ..< languageModel.configuration.hiddenLayers {
            let prefix = "language_model.model.layers.\(l).mlp"

            // Format A
            let gateUpKey = "\(prefix).experts.gate_up_proj"
            if let gateUp = newWeights[gateUpKey] {
                newWeights[gateUpKey] = nil
                let mid = gateUp.dim(-2) / 2
                newWeights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[.ellipsis, ..<mid, 0...]
                newWeights["\(prefix).switch_mlp.up_proj.weight"]   = gateUp[.ellipsis, mid..., 0...]
                if let dp = newWeights["\(prefix).experts.down_proj"] {
                    newWeights["\(prefix).experts.down_proj"] = nil
                    newWeights["\(prefix).switch_mlp.down_proj.weight"] = dp
                }
            }

            // Format B
            if newWeights["\(prefix).experts.0.gate_proj.weight"] != nil {
                for projName in ["gate_proj", "up_proj", "down_proj"] {
                    let perExpert = (0 ..< nExperts).compactMap {
                        newWeights["\(prefix).experts.\($0).\(projName).weight"]
                    }
                    let perExpertScale = (0 ..< nExperts).compactMap {
                        newWeights["\(prefix).experts.\($0).\(projName).weight_scale_inv"]
                    }

                    if perExpert.count == nExperts {
                        if perExpertScale.count == nExperts {
                            // FP8 checkpoint: eager per-expert dequant at load time.
                            // Avoids re-running fromFp8 + block-scale on the full [256,outDim,inDim]
                            // stacked tensor on every forward pass (would be prohibitively slow).
                            let bs = 128
                            let dequanted: [MLXArray] = zip(perExpert, perExpertScale).map { w, inv in
                                let wFp = MLXFast.fromFp8(w, dtype: .bfloat16)
                                let (m, n) = (wFp.dim(0), wFp.dim(1))
                                let padB = (bs - m % bs) % bs
                                let padS = (bs - n % bs) % bs
                                var p = MLX.padded(wFp, widths: [[0, padB], [0, padS]])
                                p = p.reshaped([(m + padB) / bs, bs, (n + padS) / bs, bs])
                                let scaled = p * inv[0..., .newAxis, 0..., .newAxis]
                                return scaled.reshaped([m + padB, n + padS])[0 ..< m, 0 ..< n].asType(.bfloat16)
                            }
                            let stacked = MLX.stacked(dequanted)
                            // Eagerly eval to pay the dequant cost at load time, not during prefill.
                            // Without this, the entire lazy graph materializes on first forward pass.
                            MLX.eval(stacked)
                            newWeights["\(prefix).switch_mlp.\(projName).weight"] = stacked
                            // Scale tensors consumed — do NOT store weight_scale_inv
                            for i in 0 ..< nExperts {
                                newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                                newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight_scale_inv")
                            }
                        } else {
                            // BF16 checkpoint: stack as-is
                            newWeights["\(prefix).switch_mlp.\(projName).weight"] = MLX.stacked(perExpert)
                            for i in 0 ..< nExperts {
                                newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                            }
                        }
                    }
                }
            }
        }

        // ── Step 4: MoE expert weight stacking (MTP heads) ──
        for l in 0 ..< languageModel.configuration.numNextnPredictLayers {
            let prefixes = [
                "language_model.mtp.\(l).layers.0.mlp",
                "language_model.mtp.layers.0.mlp",
                "language_model.mtp.layers.\(l).mlp"
            ]
            for prefix in prefixes {
                // Format A
                let gateUpKey = "\(prefix).experts.gate_up_proj"
                if let gateUp = newWeights[gateUpKey] {
                    newWeights[gateUpKey] = nil
                    let mid = gateUp.dim(-2) / 2
                    newWeights["\(prefix).switch_mlp.gate_proj.weight"] = gateUp[.ellipsis, ..<mid, 0...]
                    newWeights["\(prefix).switch_mlp.up_proj.weight"]   = gateUp[.ellipsis, mid..., 0...]
                    if let dp = newWeights["\(prefix).experts.down_proj"] {
                        newWeights["\(prefix).experts.down_proj"] = nil
                        newWeights["\(prefix).switch_mlp.down_proj.weight"] = dp
                    }
                }

                // Format B
                if newWeights["\(prefix).experts.0.gate_proj.weight"] != nil {
                    for projName in ["gate_proj", "up_proj", "down_proj"] {
                        let perExpert = (0 ..< nExperts).compactMap {
                            newWeights["\(prefix).experts.\($0).\(projName).weight"]
                        }
                        let perExpertScale = (0 ..< nExperts).compactMap {
                            newWeights["\(prefix).experts.\($0).\(projName).weight_scale_inv"]
                        }
                        if perExpert.count == nExperts {
                            if perExpertScale.count == nExperts {
                                let bs = 128
                                let dequanted: [MLXArray] = zip(perExpert, perExpertScale).map { w, inv in
                                    let wFp = MLXFast.fromFp8(w, dtype: .bfloat16)
                                    let (m, n) = (wFp.dim(0), wFp.dim(1))
                                    let padB = (bs - m % bs) % bs; let padS = (bs - n % bs) % bs
                                    var p = MLX.padded(wFp, widths: [[0, padB], [0, padS]])
                                    p = p.reshaped([(m + padB) / bs, bs, (n + padS) / bs, bs])
                                    return (p * inv[0..., .newAxis, 0..., .newAxis]).reshaped([m + padB, n + padS])[0 ..< m, 0 ..< n].asType(.bfloat16)
                                }
                                let stacked = MLX.stacked(dequanted)
                                MLX.eval(stacked)
                                newWeights["\(prefix).switch_mlp.\(projName).weight"] = stacked
                                for i in 0 ..< nExperts {
                                    newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                                    newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight_scale_inv")
                                }
                            } else {
                                newWeights["\(prefix).switch_mlp.\(projName).weight"] = MLX.stacked(perExpert)
                                for i in 0 ..< nExperts {
                                    newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                                }
                            }
                        }
                    }
                }
            }
        }

        // ── Step 5: Eager FP8 block-wise dequantization for remaining non-expert Linear layers ──
        // After Steps 3+4, ALL switch_mlp expert scale tensors have been consumed during stacking.
        // Any remaining "weight_scale_inv" keys belong to regular Linear layers
        // (attention projections, shared_expert, GatedDeltaNet, lm_head, etc.).
        // These cannot carry weight_scale_inv, so we eagerly dequantize here.
        var processed = [String: MLXArray]()
        for (key, value) in newWeights {
            if key.hasSuffix(".weight_scale_inv") {
                let wKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let w = newWeights[wKey], processed[wKey] == nil {
                    // Swift MLX maps F8_E4M3 → uint8; fromFp8 gives proper signed floats.
                    let wFp: MLXArray = MLXFast.fromFp8(w, dtype: .bfloat16)
                    let bs = 128
                    let (m, n) = (wFp.dim(0), wFp.dim(1))
                    let padBottom = (bs - m % bs) % bs
                    let padSide   = (bs - n % bs) % bs
                    var padded = MLX.padded(wFp, widths: [[0, padBottom], [0, padSide]])
                    padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
                    let scaled = padded * value[0..., .newAxis, 0..., .newAxis]
                    let dequant = scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                    processed[wKey] = dequant.asType(.bfloat16)
                }
                // Drop the scale tensor — Linear has no slot for it.
            } else if processed[key] == nil {
                processed[key] = value
            }
        }
        if !processed.isEmpty { newWeights = processed }


        return languageModel.sanitize(weights: newWeights)
    }

}
