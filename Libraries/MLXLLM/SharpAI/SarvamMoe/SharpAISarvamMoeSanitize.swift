//
//  SharpAISarvamMoeSanitize.swift
//  MLXLLM
//
//  Maps the raw sarvam_moe checkpoint layout onto the module tree:
//  drops rotary caches, MTP or extra layers past num_hidden_layers, and the
//  lm_head when embeddings are tied. Per-expert tensors
//  `mlp.experts.{e}.{proj}.{suffix}` are stacked into `mlp.switch_mlp.{proj}.{suffix}`.
//  Layout checked against sarvamai/sarvam-30b's model.safetensors.index.json: 19 layers,
//  layer 0 dense, layers 1-18 with 128 experts each, no extra MTP layer.
//

import Foundation
import MLX
import MLXLMCommon

enum SharpAISarvamMoeSanitize {

    static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAISarvamMoeConfiguration
    ) -> [String: MLXArray] {
        var sanitized = filterLMHeadWeights(
            from: weights, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        let layerCount = configuration.hiddenLayers
        sanitized = sanitized.filter { key, _ in
            if key.contains("rotary_emb.inv_freq") { return false }
            guard let index = layerIndex(of: key) else { return true }
            return index < layerCount
        }

        for layer in 0 ..< layerCount where layer >= configuration.firstKDenseReplace {
            let prefix = "model.layers.\(layer)"
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).mlp.experts.0.\(proj).\(suffix)"] != nil else {
                        continue
                    }
                    let parts = (0 ..< configuration.numExperts).compactMap { e in
                        sanitized.removeValue(forKey: "\(prefix).mlp.experts.\(e).\(proj).\(suffix)")
                    }
                    sanitized["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = stacked(parts)
                }
            }
        }

        return sanitized
    }

    /// Layer index for keys like `model.layers.12.x`, nil otherwise.
    static func layerIndex(of key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        return Int(key.dropFirst(prefix.count).prefix { $0.isNumber })
    }
}
