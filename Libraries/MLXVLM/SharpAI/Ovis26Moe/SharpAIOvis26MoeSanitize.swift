// SharpAI fork overlay: ovis2_6_moe checkpoint key sanitization. Not part of upstream.
//
// The prefixes are checked against the real AIDC-AI/Ovis2.6-30B-A3B model.safetensors.index.json:
// all 19,306 keys fall under llm. (18,867), visual_tokenizer. (438: vit and head) or vte.weight (1).
// Visual weights are dropped until the visual path exists.

import Foundation
import MLX

public enum SharpAIOvis26MoeSanitizer {

    static let languagePrefix = "llm."
    static let visualPrefixes = ["visual_tokenizer.", "vte."]

    /// Drops visual weights and strips the language prefix from text weights.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if visualPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var newKey = key
            if newKey.hasPrefix(languagePrefix) {
                newKey = String(newKey.dropFirst(languagePrefix.count))
            }

            guard sanitized[newKey] == nil else {
                throw SharpAIOvis26MoeError.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
