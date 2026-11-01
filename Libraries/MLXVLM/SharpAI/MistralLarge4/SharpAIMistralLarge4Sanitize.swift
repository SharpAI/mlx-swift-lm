// SharpAI fork overlay: mistral_large4 checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names. Only non-weight buffers and the tied LM head are dropped.
// UNVERIFIED: no Mistral Large 4 checkpoint index was available, so the key prefixes
// (`lm_head.weight`, `language_model.lm_head.weight`) are assumed, not confirmed.

import Foundation
import MLX

public enum SharpAIMistralLarge4Sanitizer {

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIMistralLarge4Configuration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tiedWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
            sanitized.removeValue(forKey: "language_model.lm_head.weight")
        }

        return sanitized
    }
}
