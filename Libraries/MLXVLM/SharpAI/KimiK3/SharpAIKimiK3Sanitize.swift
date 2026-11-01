// SharpAI fork overlay: kimi_k3 checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names (`language_model.model.layers.N...`, `language_model.lm_head.weight`), which is the
// module tree of SharpAIKimiK3Model. Layout from moonshotai/Kimi-K3 model.safetensors.index.json (fetched
// 2026-10-30): per-layer KDA tensors `self_attn.{q,k,v}_conv1d.weight`, MLA tensors `self_attn.{q_a,kv_a,...}`,
// per-expert `block_sparse_moe.experts.E.{w1,w2,w3}`, `shared_experts.*`, `routed_expert_*`, and the vision keys
// `vision_tower.*`, `mm_projector.*`.
//
// Changes:
// - KDA short-conv weights: q, k and v conv1d weights are merged into one depthwise conv over the concatenated
//   channels (the same channel order the forward pass splits). Torch layout [C, 1, K] becomes MLX [C, K, 1].
//
// - Routed experts: per-expert `experts.E.w1/w3/w2.{weight,scales,biases}` are stacked on a new leading axis into
//   `switch_mlp.gate_proj/up_proj/down_proj.*` (w1 is the gate, w3 the up and w2 the down projection).
//
// - Vision keys: `vision_tower.*` and `mm_projector.*` are dropped, because the vision tower is not ported and
//   the text-only module tree has nothing to receive them.
//
// Not done (KNOWN GAP, see the remaining list in the port notes):
// - The released experts are MXFP4 packed: `experts.E.w{1,2,3}.weight_packed` (uint8) with
//   `weight_scale` (uint8 E8M0, group 32). They pass through unchanged and will not load into the fp layers.
//   The expert tensors also need the mxfp4 mapping in SharpAIOverlay.mapMXFP4Experts before quantized layers exist.

import Foundation
import MLX

public enum SharpAIKimiK3Sanitizer {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = weights.filter { key, _ in
            !key.hasPrefix("vision_tower.") && !key.hasPrefix("mm_projector.")
        }
        for key in Array(out.keys) where key.hasSuffix(".self_attn.q_conv1d.weight") {
            let base = String(key.dropLast("q_conv1d.weight".count))
            let splitKeys = ["q", "k", "v"].map { "\(base)\($0)_conv1d.weight" }
            guard splitKeys.allSatisfy({ out[$0] != nil }), out["\(base)conv1d.weight"] == nil else {
                continue
            }
            let channelFirst = splitKeys.map { sharpAIKimiK3ConvLayout(out.removeValue(forKey: $0)!) }
            out["\(base)conv1d.weight"] = MLX.concatenated(channelFirst, axis: 0)
        }
        return stackExperts(out)
    }

    /// Stacks `<prefix>.experts.E.wN.<suffix>` into `<prefix>.switch_mlp.<proj>.<suffix>` when the experts
    /// 0 ..< n are all present. Other keys (including packed MXFP4 tensors) are left unchanged.
    private static func stackExperts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let projections = ["w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"]
        let suffixes: Set<String> = ["weight", "scales", "biases"]
        var groups: [String: [Int: String]] = [:]  // stacked key -> expert index -> source key
        for key in weights.keys {
            let parts = key.components(separatedBy: ".")
            guard let marker = parts.lastIndex(of: "experts"), marker + 3 < parts.count,
                parts.count == marker + 4, let expert = Int(parts[marker + 1]),
                let projection = projections[parts[marker + 2]], suffixes.contains(parts[marker + 3])
            else { continue }
            let stacked =
                (parts[..<marker] + ["switch_mlp", projection, parts[marker + 3]])
                .joined(separator: ".")
            groups[stacked, default: [:]][expert] = key
        }
        var out = weights
        for (stacked, sources) in groups {
            let count = sources.count
            guard (0 ..< count).allSatisfy({ sources[$0] != nil }) else { continue }
            let arrays = (0 ..< count).map { out[sources[$0]!]! }
            for index in 0 ..< count { out.removeValue(forKey: sources[index]!) }
            out[stacked] = MLX.stacked(arrays, axis: 0)
        }
        return out
    }
}

/// Torch depthwise conv weights are `[C, 1, K]` (or `[C, K]`). MLX wants `[C, K, 1]`.
private func sharpAIKimiK3ConvLayout(_ weight: MLXArray) -> MLXArray {
    if weight.ndim == 2 {
        return MLX.expandedDimensions(weight, axis: 2)
    }
    if weight.ndim == 3, weight.dim(1) == 1 {
        return weight.movedAxis(source: 2, destination: 1)
    }
    return weight
}
