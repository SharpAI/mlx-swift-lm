// SharpAI fork overlay: olmo_hybrid checkpoint key mapping. Not part of upstream mlx-swift-lm.
//
// The module tree uses the HF checkpoint names, so only three things change:
// drop mtp.* and rotary buffers (HF ignores them on load), drop lm_head when embeddings are tied,
// and move the torch depthwise conv weight [C, 1, K] to the MLX layout [C, K, 1].

import Foundation
import MLX
import MLXLMCommon

public enum SharpAIOlmoHybridSanitize {

    public static func sanitize(
        _ weights: [String: MLXArray], tiedWordEmbeddings: Bool
    ) -> [String: MLXArray] {
        var out = filterLMHeadWeights(
            from: weights.filter { key, _ in
                !key.hasPrefix("mtp.") && !key.contains("rotary_emb.inv_freq")
            },
            tiedWordEmbeddings: tiedWordEmbeddings)

        for key in Array(out.keys) where key.hasSuffix("linear_attn.conv1d.weight") {
            guard let weight = out[key], weight.ndim == 3, weight.dim(-1) != 1 else { continue }
            out[key] = weight.movedAxis(source: 2, destination: 1)
        }

        return out
    }
}
