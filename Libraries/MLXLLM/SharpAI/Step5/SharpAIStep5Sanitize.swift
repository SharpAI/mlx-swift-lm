//
//  SharpAIStep5Sanitize.swift
//  MLXLLM
//
//  Checkpoint key cleanup for `step5`. Not part of upstream mlx-swift-lm.
//  Key names come from stepfun-ai/Step-5-Preview-BF16 model.safetensors.index.json (sha256
//  bc5275e9...). Expert tensors are already stacked as [experts, out, in] under
//  `moe.{gate,up,down}_proj.weight`. They are renamed to the `moe.switch_mlp.*` tree that
//  SwitchGLU uses. Everything else keeps its Hugging Face name.
//

import Foundation
import MLX

public enum SharpAIStep5Sanitize {

    /// Drops precomputed rotary tables. Other keys pass through unchanged.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }

    /// Config-aware cleanup for the text model. Drops rotary buffers, vision and projector
    /// keys, MTP layers (index >= num_hidden_layers), and the DSA indexer tensors that
    /// this port does not build. It also renames the stacked routed experts to switch_mlp.
    static func sanitize(
        weights: [String: MLXArray], configuration: SharpAIStep5Configuration
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            if key.hasPrefix("vision_model.") || key.hasPrefix("vit_") { continue }
            if let layer = layerIndex(of: key), layer >= configuration.hiddenLayers { continue }
            if key.contains(".self_attn.sparse_indexer_") || key.hasSuffix(".self_attn.ssmax_s") {
                continue
            }
            if let renamed = stackedExpertKey(key) {
                out[renamed] = value
                continue
            }
            out[key] = value
        }
        return out
    }

    private static func layerIndex(of key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        let rest = key.dropFirst(prefix.count)
        guard let end = rest.firstIndex(of: ".") else { return nil }
        return Int(rest[..<end])
    }

    private static func stackedExpertKey(_ key: String) -> String? {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let suffix = ".moe.\(projection).weight"
            if key.hasSuffix(suffix) {
                return String(key.dropLast(suffix.count)) + ".moe.switch_mlp.\(projection).weight"
            }
        }
        return nil
    }
}
