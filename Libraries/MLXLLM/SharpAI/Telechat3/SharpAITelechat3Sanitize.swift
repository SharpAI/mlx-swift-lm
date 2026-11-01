// SharpAI fork overlay: telechat3 checkpoint key handling. Not part of upstream mlx-swift-lm.
//
// The keys match LlamaModel, so the default table is empty.
// UNVERIFIED: no telechat3 checkpoint is in the repo. Confirm the tensor names before adding rows.

import Foundation
import MLX

public enum SharpAITelechat3Sanitizer {

    /// HF checkpoint prefix -> module prefix. Empty until telechat3 tensor names are confirmed.
    public static let keyMapping: [(from: String, to: String)] = []

    /// Drops rotary tables and tied `lm_head` weights, then applies the first prefix rewrite.
    public static func sanitize(
        _ weights: [String: MLXArray],
        tieWordEmbeddings: Bool,
        mapping: [(from: String, to: String)] = keyMapping
    ) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        out.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("self_attn.rotary_emb.inv_freq") { continue }
            if tieWordEmbeddings, key.split(separator: ".").contains("lm_head") { continue }

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
