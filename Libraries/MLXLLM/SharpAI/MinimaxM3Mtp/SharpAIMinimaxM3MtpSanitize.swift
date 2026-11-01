// SharpAI fork overlay: minimax_m3_mtp checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps only the MTP layer tensors and renames them to the head's keys, per vLLM
// `MiniMaxM3MTP._map_checkpoint_name` (vllm/models/minimax_m3/amd/mtp.py):
//   - a leading `language_model.` prefix (bundled checkpoints) is stripped
//   - `model.mtp.layers.<i>.X` becomes `model.layers.<i>.X`
//   - `weight_scale_inv` becomes `weight_scale` (block scales, as vLLM does)
//   - every other key is dropped. vLLM ignores keys that belong to other modules.
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
        for (name, value) in weights {
            guard let mapped = mapKey(name) else { continue }
            out[mapped] = value
        }
        return out
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
