// SharpAI fork overlay: hrm_text checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps HF key names. Only drops keys the Swift model does not load.
// UNVERIFIED: the checkpoint layout for hrm_text is not known. Confirm against a real checkpoint.

import Foundation
import MLX

public enum SharpAIHrmTextSanitizer {

    /// Drops the tied `lm_head` weight when embeddings are tied. Other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIHrmTextConfiguration
    ) -> [String: MLXArray] {
        weights.filter { key, _ in
            if configuration.tieWordEmbeddings,
                key.split(separator: ".").contains("lm_head")
            {
                return false
            }
            return true
        }
    }
}
