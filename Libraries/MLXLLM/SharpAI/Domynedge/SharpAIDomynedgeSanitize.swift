// SharpAI fork overlay: domynedge checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names. This drops buffers the loader does not use, and the
// lm_head tensors when embeddings are tied.
// UNVERIFIED: the checkpoint may store a lm_head despite tie_word_embeddings. Dropping it
// is then still correct, because the output projection reuses embed_tokens.

import Foundation
import MLX

public enum SharpAIDomynedgeSanitizer {

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIDomynedgeConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { !$0.key.hasPrefix("lm_head.") }
        }

        return sanitized
    }
}
