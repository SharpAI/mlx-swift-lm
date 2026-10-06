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

    override public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        // ── Step 1: FP8 dequantization (official Qwen3.6-35B-A3B-FP8 checkpoint) ──
        // The FP8 release stores quantized weights alongside weight_scale_inv tensors.
        // We preserve them and stack them so they can be lazily dequantized in SwitchLinear.
        // ── Step 2: Key remapping ──
        let checkpoint = try Qwen35CheckpointPolicy.prepareTarget(
            .init(weights: weights), layout: .wrappedText,
            tiedWordEmbeddings: languageModel.configuration.tieWordEmbeddings,
            retainMTP: MTPConfig.retainMTPWeights)
        var newWeights = checkpoint.weights

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
                let isStreaming = ExpertStreamingConfig.shared.isEnabled
                for projName in ["gate_proj", "up_proj", "down_proj"] {
                    let perExpert = (0 ..< nExperts).compactMap {
                        newWeights["\(prefix).experts.\($0).\(projName).weight"]
                    }
                    let perExpertScale = (0 ..< nExperts).compactMap {
                        newWeights["\(prefix).experts.\($0).\(projName).weight_scale_inv"]
                    }

                    if perExpert.count == nExperts {
                        if perExpertScale.count == nExperts {
                            let stackedScales = MLX.stacked(perExpertScale)
                            MLX.eval(stackedScales)
                            newWeights["\(prefix).switch_mlp.\(projName).weight_scale_inv"] = stackedScales
                            
                            if !isStreaming {
                                let stackedWeights = MLX.stacked(perExpert)
                                MLX.eval(stackedWeights)
                                newWeights["\(prefix).switch_mlp.\(projName).weight"] = stackedWeights
                            }
                            
                            for i in 0 ..< nExperts {
                                newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                                newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight_scale_inv")
                            }
                        } else {
                            if !isStreaming {
                                newWeights["\(prefix).switch_mlp.\(projName).weight"] = MLX.stacked(perExpert)
                            }
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
                    let isStreaming = ExpertStreamingConfig.shared.isEnabled
                    for projName in ["gate_proj", "up_proj", "down_proj"] {
                        let perExpert = (0 ..< nExperts).compactMap {
                            newWeights["\(prefix).experts.\($0).\(projName).weight"]
                        }
                        let perExpertScale = (0 ..< nExperts).compactMap {
                            newWeights["\(prefix).experts.\($0).\(projName).weight_scale_inv"]
                        }

                        if perExpert.count == nExperts {
                            if perExpertScale.count == nExperts {
                                let stackedScales = MLX.stacked(perExpertScale)
                                MLX.eval(stackedScales)
                                newWeights["\(prefix).switch_mlp.\(projName).weight_scale_inv"] = stackedScales
                                
                                if !isStreaming {
                                    let stackedWeights = MLX.stacked(perExpert)
                                    MLX.eval(stackedWeights)
                                    newWeights["\(prefix).switch_mlp.\(projName).weight"] = stackedWeights
                                }
                                
                                for i in 0 ..< nExperts {
                                    newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight")
                                    newWeights.removeValue(forKey: "\(prefix).experts.\(i).\(projName).weight_scale_inv")
                                }
                            } else {
                                if !isStreaming {
                                    newWeights["\(prefix).switch_mlp.\(projName).weight"] = MLX.stacked(perExpert)
                                }
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
        let keys = Array(newWeights.keys)
        for key in keys {
            if key.hasSuffix(".weight_scale_inv") {
                if key.contains(".switch_mlp.") {
                    continue
                }
                let wKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let w = newWeights[wKey], let scale = newWeights[key] {
                    // Aggressively free the source references before eval
                    newWeights.removeValue(forKey: wKey)
                    newWeights.removeValue(forKey: key)
                    
                    let wFp: MLXArray = MLXFast.fromFp8(w, dtype: .bfloat16)
                    let bs = 128
                    let (m, n) = (wFp.dim(0), wFp.dim(1))

                    let padBottom = (bs - m % bs) % bs
                    let padSide   = (bs - n % bs) % bs
                    var padded = MLX.padded(wFp, widths: [[0, padBottom], [0, padSide]])
                    padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
                    let scaled = padded * scale[0..., .newAxis, 0..., .newAxis]
                    let dequant = scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
                    
                    let evaluated = dequant.asType(.bfloat16)
                    MLX.eval(evaluated)
                    newWeights[wKey] = evaluated
                }
            }
        }

        return languageModel.sanitize(weights: newWeights)
    }

}
