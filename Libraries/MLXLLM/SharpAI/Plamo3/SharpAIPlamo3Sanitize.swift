// SharpAI fork overlay: plamo3 checkpoint key remapping. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the plamo3 tensor names are not in the repo or in local sources (gated repo).
// The default table is empty, so keys pass through unchanged. The helper applies a prefix table
// once the real names are confirmed.

import Foundation
import MLX

public enum SharpAIPlamo3Sanitizer {

    /// HF checkpoint prefix -> module prefix. Empty until the plamo3 tensor names are confirmed (UNVERIFIED).
    public static let keyMapping: [(from: String, to: String)] = []

    /// Applies the first matching prefix rewrite to each key. With an empty table this is a no-op.
    public static func sanitize(
        _ weights: [String: MLXArray],
        mapping: [(from: String, to: String)] = keyMapping
    ) -> [String: MLXArray] {
        guard !mapping.isEmpty else { return weights }
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            for (from, to) in mapping where newKey.hasPrefix(from) {
                newKey = to + String(newKey.dropFirst(from.count))
                break
            }
            out[newKey] = value
        }
        return out
    }
}
