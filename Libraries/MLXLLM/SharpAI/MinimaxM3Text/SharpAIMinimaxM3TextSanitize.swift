// SharpAI fork overlay: minimax_m3_text checkpoint key remapping. Not part of upstream mlx-swift-lm.
//
// `sanitize(_:mapping:)` applies a prefix table and is a no-op while the table is empty.
// `remapCheckpoint` maps the layouts the model needs onto the module tree:
//   - fused routed experts `mlp.experts.gate_up_proj` [E, 2I, H] are split into gate and up;
//     `mlp.experts.down_proj` [E, H, I] is kept as-is;
//   - per-expert tensors `mlp.experts.<i>.{gate_proj,up_proj,down_proj}` or `{w1,w3,w2}` are stacked;
//   - fused dense or shared `mlp.gate_up_proj` [2I, H] is split into gate and up;
//   - the router selection bias is moved to `mlp.gate.e_score_correction_bias`;
//   - MTP tensors, rotary buffers, and the tied `lm_head.weight` are dropped.
// UNVERIFIED: the M3 checkpoint tensor names are not in the repo or in local sources. The layouts
// above are the ones the model accepts. None has been checked against a real index file.

import Foundation
import MLX

public enum SharpAIMinimaxM3TextSanitizer {

    /// HF checkpoint prefix -> module prefix. Empty until the M3 tensor names are confirmed (UNVERIFIED).
    public static let keyMapping: [(from: String, to: String)] = []

    /// Maps a checkpoint onto this model's module keys. See the file header for the layouts handled.
    public static func remapCheckpoint(
        _ weights: [String: MLXArray], configuration: SharpAIMinimaxM3TextConfiguration
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        // module prefix + stacked name -> expert index -> tensor
        var expertParts = [String: [Int: MLXArray]]()

        for (key, value) in weights {
            if key.hasPrefix("mtp") || key.contains(".mtp.") || key.contains("rotary_emb") {
                continue
            }
            if let layer = decoderLayerIndex(key), layer >= configuration.hiddenLayers {
                continue
            }
            if configuration.tieWordEmbeddings, key == "lm_head.weight" {
                continue
            }

            var name = key
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

    /// Layer index for `model.layers.<n>.…` keys, nil otherwise.
    private static func decoderLayerIndex(_ key: String) -> Int? {
        let prefix = "model.layers."
        guard key.hasPrefix(prefix) else { return nil }
        return Int(key.dropFirst(prefix.count).prefix { $0 != "." })
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

    /// Applies the first matching prefix rewrite to each key. With an empty table this is a no-op.
    public static func sanitize(
        _ weights: [String: MLXArray],
        mapping: [(from: String, to: String)] = keyMapping
    ) -> [String: MLXArray] {
        guard !mapping.isEmpty else { return weights }
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            for (from, to) in mapping where newKey.hasPrefix(from) {
                newKey = to + String(newKey.dropFirst(from.count))
                break
            }
            out[newKey] = value
        }
        return out
    }
}
