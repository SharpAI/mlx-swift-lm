// SharpAI fork overlay: bailing_hybrid checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps HF key names, drops keys the Swift model does not load, stacks per-expert MoE weights into
// switch_mlp and splits kv_b_proj into embed_q / unembed_out.
// UNVERIFIED: MTP/nextn layers are assumed to sit at indices num_hidden_layers..<num_hidden_layers + num_nextn_predict_layers.
// Keys named differently are kept, not dropped.
// Confirm the key names and the kv_b_proj layout against a real checkpoint.

import Foundation
import MLX

public enum SharpAIBailingHybridSanitizer {

    /// Drops tied `lm_head` weights and MTP/nextn layer weights, stacks `mlp.experts.N.*` into
    /// `mlp.switch_mlp.*` and splits `attention.kv_b_proj.weight` into `embed_q` / `unembed_out`.
    /// Other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIBailingHybridConfiguration
    ) -> [String: MLXArray] {
        var out = weights.filter { key, _ in
            if configuration.tieWordEmbeddings,
                key.split(separator: ".").contains("lm_head")
            {
                return false
            }
            if let index = layerIndex(of: key),
                index >= configuration.hiddenLayers,
                index < configuration.hiddenLayers + configuration.numNextnPredictLayers
            {
                return false
            }
            return true
        }

        let heads = configuration.attentionHeads
        let nope = configuration.qkNopeHeadDim ?? 0
        for l in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(l)"

            if let experts = configuration.numExperts, experts > 0 {
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases", "bias"] {
                        let expertKeys = (0 ..< experts).map {
                            "\(prefix).mlp.experts.\($0).\(proj).\(suffix)"
                        }
                        guard expertKeys.allSatisfy({ out[$0] != nil }) else { continue }
                        let parts = expertKeys.map { out.removeValue(forKey: $0)! }
                        out["\(prefix).mlp.switch_mlp.\(proj).\(suffix)"] = stacked(parts)
                    }
                }
            }

            // kv_b_proj is [heads * (nope + v), kvLora]; embed_q takes the key rows transposed.
            let kvBKey = "\(prefix).attention.kv_b_proj.weight"
            if heads > 0, let kvB = out[kvBKey], kvB.ndim == 2, kvB.dim(0) % heads == 0 {
                let perHead = kvB.reshaped(heads, kvB.dim(0) / heads, kvB.dim(1))
                out["\(prefix).attention.embed_q.weight"] =
                    perHead[0..., ..<nope, 0...].swappedAxes(-1, -2)
                out["\(prefix).attention.unembed_out.weight"] = perHead[0..., nope..., 0...]
                out.removeValue(forKey: kvBKey)
            }
        }
        return out
    }

    /// Returns N for keys shaped like `...layers.N.<rest>`, or nil.
    public static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else {
            return nil
        }
        return Int(parts[at + 1])
    }
}
