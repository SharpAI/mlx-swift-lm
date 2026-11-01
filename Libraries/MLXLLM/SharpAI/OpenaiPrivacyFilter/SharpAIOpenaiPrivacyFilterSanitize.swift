// SharpAI fork overlay: openai_privacy_filter checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// The HF checkpoint stores each expert projection as one tensor in [experts, in, out] layout.
// SwitchLinear expects [experts, out, in], so those tensors are transposed. The names are
// mapped to the SwitchLinear parameters: "gate_up_proj" to "gate_up_proj.weight", and the same
// for the bias.
// Only model.* and score.* keys are kept. Rotary buffers and any other top-level tensors are
// dropped.

import Foundation
import MLX

enum SharpAIOpenaiPrivacyFilterSanitizer {

    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasSuffix(".mlp.experts.gate_up_proj")
                || key.hasSuffix(".mlp.experts.down_proj")
            {
                sanitized[key + ".weight"] = value.swappedAxes(1, 2)
            } else if key.hasSuffix(".mlp.experts.gate_up_proj_bias")
                || key.hasSuffix(".mlp.experts.down_proj_bias")
            {
                sanitized[String(key.dropLast("_bias".count)) + ".bias"] = value
            } else if isKept(key) {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    static func isKept(_ key: String) -> Bool {
        if key.contains("rotary_emb") || key.contains("inv_freq") {
            return false
        }
        return key.hasPrefix("model.") || key.hasPrefix("score.")
    }
}
