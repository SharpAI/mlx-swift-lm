// SharpAI fork overlay: minimax_m3_mtp checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps only the MTP layer tensors and renames them to the head's keys, per vLLM
// `MiniMaxM3MTP._map_checkpoint_name` (vllm/models/minimax_m3/amd/mtp.py):
//   - a leading `language_model.` prefix (bundled checkpoints) is stripped
//   - `model.mtp.layers.<i>.X` becomes `model.layers.<i>.X`
//   - `weight_scale_inv` becomes `weight_scale` (block scales, as vLLM does)
//   - every other key is dropped. vLLM ignores keys that belong to other modules.
//   - the transformer layer's MoE tensors are brought onto the head's module tree the way the text
//     sanitizer does for the decoder layers: fused `experts.gate_up_proj` is split, per-expert
//     tensors `experts.<i>.{gate_proj,up_proj,down_proj}` or `{w1,w3,w2}` are stacked, fused
//     `mlp.gate_up_proj` is split, and the router bias moves to `mlp.gate.e_score_correction_bias`
// Deviation: vLLM also loads `model.embed_tokens.weight` and `lm_head.weight` into the MTP module.
// This head takes the target's embedding and lm_head, so those two keys are dropped here.
//
// Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/minimax_m3/amd/mtp.py

import Foundation
import MLX

public enum SharpAIMinimaxM3MtpSanitizer {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        // module prefix + stacked name -> expert index -> tensor
        var expertParts = [String: [Int: MLXArray]]()

        for (key, value) in weights {
            guard let mapped = mapKey(key) else { continue }

            var name = mapped
            if name.hasSuffix(".experts.e_score_correction_bias") {
                name = String(name.dropLast(".experts.e_score_correction_bias".count))
                    + ".gate.e_score_correction_bias"
            } else if name.hasSuffix(".mlp.e_score_correction_bias") {
                name = String(name.dropLast(".e_score_correction_bias".count))
                    + ".gate.e_score_correction_bias"
            }

            let stripped =
                name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name

            if value.ndim == 3, stripped.hasSuffix(".experts.gate_up_proj") {
                let base = String(stripped.dropLast("gate_up_proj".count))
                let halves = split(value, parts: 2, axis: 1)
                out[base + "gate_proj.weight"] = halves[0]
                out[base + "up_proj.weight"] = halves[1]
            } else if value.ndim == 3, stripped.hasSuffix(".experts.down_proj") {
                out[stripped + ".weight"] = value
            } else if value.ndim == 2, stripped.hasSuffix(".gate_up_proj") {
                let base = String(stripped.dropLast("gate_up_proj".count))
                let halves = split(value, parts: 2, axis: 0)
                out[base + "gate_proj.weight"] = halves[0]
                out[base + "up_proj.weight"] = halves[1]
            } else if let parsed = expertKey(stripped) {
                expertParts[parsed.prefix + parsed.name + ".weight", default: [:]][parsed.index] =
                    value
            } else {
                out[name] = value
            }
        }

        for (key, byIndex) in expertParts {
            out[key] = stacked(byIndex.keys.sorted().map { byIndex[$0]! })
        }
        return out
    }

    /// Splits `…mlp.experts.<i>.<name>` (name without `.weight`) into the module prefix, the
    /// expert index, and the stacked name (gate_proj, up_proj or down_proj).
    private static func expertKey(_ key: String) -> (prefix: String, index: Int, name: String)? {
        guard let range = key.range(of: ".experts.") else { return nil }
        let parts = key[range.upperBound...].split(separator: ".")
        guard parts.count == 2, let index = Int(parts[0]) else { return nil }
        let name: String
        switch parts[1] {
        case "gate_proj", "w1": name = "gate_proj"
        case "up_proj", "w3": name = "up_proj"
        case "down_proj", "w2": name = "down_proj"
        default: return nil
        }
        return (String(key[..<range.upperBound]), index, name)
    }

    /// Returns the head's key for a checkpoint key, or nil when the key is not an MTP layer tensor.
    static func mapKey(_ name: String) -> String? {
        let prefix = "language_model."
        let stripped = name.hasPrefix(prefix) ? String(name.dropFirst(prefix.count)) : name
        guard stripped.hasPrefix("model.mtp.layers.") else { return nil }
        var mapped = stripped.replacingOccurrences(
            of: "model.mtp.layers.", with: "model.layers.")
        if mapped.hasSuffix("weight_scale_inv") {
            mapped = String(mapped.dropLast("weight_scale_inv".count)) + "weight_scale"
        }
        return mapped
    }
}
