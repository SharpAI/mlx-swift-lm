// SharpAI fork overlay: alpamayo2_super checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Key prefixes verified against https://huggingface.co/nvidia/Alpamayo2-Super/blob/main/model.safetensors.index.json:
// keys start with `expert.` and `vlm.` (for example vlm.model.language_model.layers.N, vlm.lm_head).
// The default prefix table is still empty; the Swift module layout that would need remapping is not written.

import Foundation
import MLX

public enum SharpAIAlpamayo2SuperSanitizer {

    /// Ordered `(from, to)` prefix rewrites. The first matching prefix wins.
    public static let defaultPrefixRemaps: [(from: String, to: String)] = []

    /// Drops rotary buffers and rewrites key prefixes. Other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray],
        prefixRemaps: [(from: String, to: String)] = defaultPrefixRemaps
    ) throws -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            let newKey = remappedKey(key, prefixRemaps: prefixRemaps)
            guard sanitized[newKey] == nil else {
                throw SharpAIAlpamayo2SuperError.duplicateWeight(newKey)
            }
            sanitized[newKey] = value
        }
        return sanitized
    }

    static func remappedKey(_ key: String, prefixRemaps: [(from: String, to: String)]) -> String {
        for (from, to) in prefixRemaps where key.hasPrefix(from) {
            return to + String(key.dropFirst(from.count))
        }
        return key
    }
}
