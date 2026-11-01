// SharpAI fork overlay: granite_swa checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Source: transformers granite_swa modeling. Keys keep their HF names: `self_attn.sinks` is a
// plain parameter, and the RoPE `inv_freq` buffers are non-persistent, so they have no keys.

import Foundation
import MLX

public enum SharpAIGraniteSwaSanitizer {

    /// Drops RoPE buffers and the tied `lm_head`. All other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIGraniteSwaConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        return sanitized
    }
}
