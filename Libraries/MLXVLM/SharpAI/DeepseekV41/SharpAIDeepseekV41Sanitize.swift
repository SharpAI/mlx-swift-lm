// SharpAI fork overlay: deepseek_v41 checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Drop rules and renames follow ml-explore/mlx-lm mlx_lm/models/deepseek_v41.py
// (Verified against: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/deepseek_v41.py).
// The published index of deepseek-ai/DeepSeek-V4.1-Flash-1017 has the raw names this file maps
// (layers.N.attn.*, hc_attn_*, ffn.experts.E.w1/w3) and 2401 mtp.* keys, which confirms the
// mtp. drop (Verified against: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/model.safetensors.index.json).
// Not written yet: the fp8/fp4 scale repack. The published checkpoint stores a `.scale` tensor
// for every weight, and those are left in place here, so the real checkpoint does not load.

import Foundation
import MLX

public enum SharpAIDeepseekV41Sanitizer {

    /// Drops rotary buffers, vision/aligner keys, MTP/DSpark heads and Engram tables.
    /// Compressor and indexer keys are kept.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter {
            !$0.key.contains("rotary_emb.inv_freq")
                && !$0.key.hasPrefix("mtp.")
                && !$0.key.hasPrefix("vision.")
                && !$0.key.hasPrefix("aligner.")
                && !$0.key.hasPrefix("image_")
                && !$0.key.contains(".engram.")
        }
    }

    /// Maps checkpoint names to the module keys of `SharpAIDeepseekV41Model`: drops, renames,
    /// and stacks the per-expert `ffn.experts.E.wK` weights into `switch_mlp` [E, out, in] tensors.
    /// A converted checkpoint (keys that already start with `model.`) is returned unchanged.
    public static func toModuleKeys(
        _ weights: [String: MLXArray], numExperts: Int
    ) -> [String: MLXArray] {
        if weights.keys.contains(where: { $0.hasPrefix("model.") }) {
            return weights
        }
        var clean: [String: MLXArray] = [:]
        // group "layer|proj" -> expert index -> (raw key, weight)
        var experts: [String: [Int: (key: String, value: MLXArray)]] = [:]
        var expertModuleKey: [String: String] = [:]
        for (key, value) in sanitize(weights) {
            if key.hasSuffix(".gate.bias_vl") {
                // bias_vl routes image tokens, which the text model never sees.
                continue
            }
            if let expert = parseExpertKey(key) {
                let group = "\(expert.layer)|\(expert.proj)"
                experts[group, default: [:]][expert.index] = (key, value)
                let projName =
                    expert.proj == "w1" ? "gate_proj" : (expert.proj == "w2" ? "down_proj" : "up_proj")
                expertModuleKey[group] =
                    "model.layers.\(expert.layer).mlp.experts.switch_mlp.\(projName).weight"
                continue
            }
            clean[renameKey(key)] = value
        }
        for (group, rows) in experts {
            guard let name = expertModuleKey[group], rows.count == numExperts else {
                // Keep the raw keys: the model then reports them as unused rather than loading
                // a partial expert stack.
                for (_, row) in rows {
                    clean[row.key] = row.value
                }
                continue
            }
            let parts = rows.keys.sorted().map { rows[$0]!.value }
            clean[name] = stacked(parts, axis: 0)
        }
        return clean
    }

    /// `layers.L.ffn.experts.E.wK.weight` -> (L, E, "wK"); nil for any other key.
    private static func parseExpertKey(_ key: String) -> (layer: Int, index: Int, proj: String)? {
        let parts = key.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 7, parts[0] == "layers", parts[2] == "ffn", parts[3] == "experts",
            parts[6] == "weight", ["w1", "w2", "w3"].contains(parts[5]),
            let layer = Int(parts[1]), let index = Int(parts[4])
        else {
            return nil
        }
        return (layer, index, parts[5])
    }

    /// Map one checkpoint name to its module name (`_rename` in mlx-lm deepseek_v41.py).
    public static func renameKey(_ key: String) -> String {
        if key == "head.weight" {
            return "lm_head.weight"
        }
        if key == "embed.weight" {
            return "model.embed_tokens.weight"
        }
        let pairs: [(String, String)] = [
            (".attn.", ".self_attn."),
            (".ffn.", ".mlp."),
            (".attn_norm.", ".input_layernorm."),
            (".ffn_norm.", ".post_attention_layernorm."),
            (".hc_attn_", ".attn_hc."),
            (".hc_ffn_", ".ffn_hc."),
            (".attn_sink", ".sinks"),
            (".wq_a.", ".q_a_proj."),
            (".wq_b.", ".q_b_proj."),
            (".wkv.", ".kv_proj."),
            (".wgate.", ".gate_proj."),
            (".wo_a.", ".o_a_proj."),
            (".wo_b.", ".o_b_proj."),
            (".q_norm.", ".q_a_norm."),
            (".compressor.norm.", ".compressor.kv_norm."),
            (".indexer.wk.", ".indexer.k_proj."),
            (".gate.bias", ".gate.e_score_correction_bias"),
            (".shared_experts.w1.", ".shared_experts.gate_proj."),
            (".shared_experts.w2.", ".shared_experts.down_proj."),
            (".shared_experts.w3.", ".shared_experts.up_proj."),
        ]
        var name = "model." + key
        for (from, to) in pairs {
            name = name.replacingOccurrences(of: from, with: to)
        }
        return name
    }
}
