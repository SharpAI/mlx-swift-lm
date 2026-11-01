// SharpAI fork overlay: granite4_vision checkpoint key sanitization. Not part of upstream.
//
// UNVERIFIED: the `model.language_model.`, `language_model.` and vision prefixes follow the
// LLaVA-style layout. Confirm them against a real checkpoint index before relying on this.
// The Q-Former keys are not mapped: no reference layout was found in the repo.
// lm_head is kept. Run GraniteModel.sanitize on the output to drop it for tied embeddings.

import Foundation
import MLX

public enum SharpAIGranite4VisionSanitizer {

    static let languagePrefixes: [(from: String, to: String)] = [
        (from: "model.language_model.", to: "model."),
        (from: "language_model.model.", to: "model."),
        (from: "language_model.lm_head.", to: "lm_head."),
    ]

    static let droppedPrefixes = [
        "model.vision_tower.",
        "vision_tower.",
        "model.multi_modal_projector.",
        "multi_modal_projector.",
    ]

    /// Drops vision-side weights and maps language keys onto the GraniteModel names.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            if droppedPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var newKey = key
            for mapping in languagePrefixes where newKey.hasPrefix(mapping.from) {
                newKey = mapping.to + String(newKey.dropFirst(mapping.from.count))
                break
            }

            guard sanitized[newKey] == nil else {
                throw SharpAIGranite4VisionError.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
