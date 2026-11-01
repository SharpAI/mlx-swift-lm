// SharpAI fork overlay: hyperclovax checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names. UNVERIFIED: post norm key names are not checked against a real index.

import Foundation
import MLX

public enum SharpAIHyperclovaxSanitizer {

    /// Drops rotary buffers and the tied `lm_head`. All other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIHyperclovaxConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        return sanitized
    }
}
