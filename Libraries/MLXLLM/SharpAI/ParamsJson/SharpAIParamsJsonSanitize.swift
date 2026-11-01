// SharpAI fork overlay: native loader for Mistral params.json checkpoints. Not part of upstream.
//
// Rewrites Mistral native weight names to the keys used by SharpAIParamsJsonEagleModel, and
// stacks per-expert w1/w2/w3 tensors into the pre-stacked switch_mlp layout.
//
// Verified against: https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-Eagle
// (consolidated.safetensors safetensors header, read by byte range: 1185 tensors). The names are
// layers.0.attention.{wq_a,wq_b,wkv_a_with_mqa,wkv_b,wo}.weight, layers.0.experts.N.{w1,w2,w3}.weight
// (no feed_forward. prefix in the real file), layers.0.shared_experts.*, layers.0.gate.weight,
// eagle_linear.weight [7168, 14336], norm.weight. The feed_forward. prefix stays accepted.
// The file has NO tok_embeddings.weight and NO output.weight: vLLM's EagleMistralLarge3ForCausalLM
// .load_weights reports model.embed_tokens.weight and lm_head.weight as loaded from the target
// model (mistral_large_3_eagle.py). The caller must share them from the target.
//
// FP8: every linear is F8_E4M3 with a per-tensor scalar `<name>.qscale_weight` (BF16). vLLM maps
// eagle_linear.qscale_weight to fc.weight_scale, a multiplicative dequant scale. qscale_act is an
// activation scale for FP8 matmul (vLLM fc.input_scale). It is dropped here because activations
// run in the model dtype.

import Foundation
import MLX
import MLXLMCommon

public enum SharpAIParamsJsonSanitize {

    static let expertProjections: [String: String] = [
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    ]

    /// Renames one per-layer key. `rest` is the part after "layers.N.".
    static func renameLayerKey(_ rest: String) -> String {
        var key = rest
        if key.hasPrefix("feed_forward.") {
            key = String(key.dropFirst("feed_forward.".count))
        }
        let table: [String: String] = [
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "attention.wq_a.weight": "self_attn.q_a_proj.weight",
            "attention.q_a_norm.weight": "self_attn.q_a_layernorm.weight",
            "attention.wq_b.weight": "self_attn.q_b_proj.weight",
            "attention.wkv_a_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "attention.kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
            "attention.wkv_b.weight": "self_attn.kv_b_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "gate.weight": "mlp.gate.weight",
            "shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
            "shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
            "shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
        ]
        return table[key] ?? key
    }

    /// Renames a top-level or per-layer key. Keys with no mapping are returned unchanged.
    static func rename(_ key: String) -> String {
        let topLevel: [String: String] = [
            "tok_embeddings.weight": "embed_tokens.weight",
            "output.weight": "lm_head.weight",
            "eagle_linear.weight": "eagle_proj.weight",
        ]
        if let mapped = topLevel[key] {
            return mapped
        }
        if key.hasPrefix("layers.") {
            let parts = key.split(separator: ".", maxSplits: 2, omittingEmptySubsequences: false)
            if parts.count == 3 {
                return "layers.\(parts[1]).\(renameLayerKey(String(parts[2])))"
            }
        }
        return key
    }

    /// Parses "[layers.N.][feed_forward.]experts.E.wK.weight".
    /// Returns the stacked target key and the expert index, or nil for other keys.
    static func parseExpertKey(_ key: String) -> (target: String, index: Int)? {
        let parts = key.split(separator: ".").map(String.init)
        let n = parts.count
        guard n >= 4, parts[n - 1] == "weight", parts[n - 4] == "experts",
            let index = Int(parts[n - 3]),
            let proj = expertProjections[parts[n - 2]]
        else {
            return nil
        }
        var prefix = Array(parts[0..<(n - 4)])
        if prefix.last == "feed_forward" {
            prefix.removeLast()
        }
        let layerPrefix = prefix.isEmpty ? "" : prefix.joined(separator: ".") + "."
        return (target: "\(layerPrefix)mlp.switch_mlp.\(proj).weight", index: index)
    }

    /// Returns the sanitized weights. Drops rotary buffers and the lm_head when embeddings are tied.
    public static func sanitize(
        _ weights: [String: MLXArray], config: SharpAIParamsJsonConfiguration
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var experts: [String: [Int: MLXArray]] = [:]

        // Pass 1: split the FP8 scale tensors from the weights. A scale is keyed by the weight it
        // dequantizes, e.g. "layers.0.experts.3.w1.qscale_weight" -> "layers.0.experts.3.w1.weight".
        var scales: [String: MLXArray] = [:]
        var plain: [(String, MLXArray)] = []
        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.") {
                key = String(key.dropFirst("model.".count))
            }
            if key.contains("rotary_emb") || key.hasSuffix("inv_freq") {
                continue
            }
            if key.hasSuffix(".qscale_act") {
                continue
            }
            if key.hasSuffix(".qscale_weight") {
                let base = String(key.dropLast(".qscale_weight".count)) + ".weight"
                scales[base] = value
                continue
            }
            plain.append((key, value))
        }

        // Pass 2: apply the per-tensor dequant scale (weight * scale, as in Mistral4), then rename.
        for (key, rawValue) in plain {
            let value = scales[key].map { rawValue * $0 } ?? rawValue
            if let parsed = parseExpertKey(key) {
                experts[parsed.target, default: [:]][parsed.index] = value
                continue
            }
            out[rename(key)] = value
        }

        for (target, byIndex) in experts {
            let ordered = byIndex.keys.sorted().compactMap { byIndex[$0] }
            out[target] = MLX.stacked(ordered, axis: 0)
        }

        if config.tiedEmbeddings {
            out.removeValue(forKey: "lm_head.weight")
        }
        return out
    }
}
