// SharpAI fork overlay: minimax_m2 key remapping. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the drafter's checkpoint tensor names are not in the repo or in local sources. The
// table is empty, so sanitize passes every key through unchanged until the names are known.

import Foundation
import MLX

public enum SharpAIMinimaxM2Sanitize {

    /// HF checkpoint prefix -> module prefix. Empty until the drafter tensor names are known (UNVERIFIED).
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
