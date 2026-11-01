// SharpAI fork overlay: hyperclovax_vision_v2 checkpoint key sanitization. Not part of upstream.
//
// UNVERIFIED: the language_model., vision_model., mm_projector. and vision_tower. prefixes are not
// in the repo. The conformance fixture has config only, no weight index. Confirm against a real
// hyperclovax_vision_v2 checkpoint index before relying on this.
// Not called by any model yet. Text keys keep their HF names after the language prefix is removed.

import Foundation
import MLX

public enum SharpAIHyperclovaxVisionV2Sanitizer {

    static let languagePrefix = "language_model."
    static let visionPrefixes = ["vision_model.", "vision_tower.", "mm_projector."]

    /// Drops vision and projector weights and strips the language prefix from text weights.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if visionPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var newKey = key
            if newKey.hasPrefix(languagePrefix) {
                newKey = String(newKey.dropFirst(languagePrefix.count))
            }

            guard sanitized[newKey] == nil else {
                throw SharpAIHyperclovaxVisionV2Error.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
