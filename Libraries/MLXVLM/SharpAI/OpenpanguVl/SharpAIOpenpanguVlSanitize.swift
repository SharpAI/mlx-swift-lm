// SharpAI fork overlay: openpangu_vl checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the repo has config.json for openPangu-VL-7B but no weight index. The "visual.",
// "model.visual." and "vision_tower." prefixes, and the "language_model." strip, are guesses from
// the Qwen2.5-VL layout. Confirm them against a real checkpoint index before relying on this.

import Foundation
import MLX

public enum SharpAIOpenpanguVlSanitizer {

    static let languagePrefix = "language_model."
    static let visionPrefixes = ["visual.", "model.visual.", "vision_tower."]

    /// Drops vision weights and strips the language prefix from text weights.
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
                throw SharpAIOpenpanguVlError.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
