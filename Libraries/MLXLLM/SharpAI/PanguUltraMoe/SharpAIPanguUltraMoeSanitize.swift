// SharpAI fork overlay: pangu_ultra_moe checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their HF names, and the model modules use the same names, so no expert stacking is
// needed. Dropped: rotary buffers, the tied LM head, and the multi-token-prediction layers. The
// published checkpoint stores the MTP layer as model.layers.<num_hidden_layers> (index 61 of
// 62 shards), and modeling_openpangu_moe.py never runs it.

import Foundation
import MLX

public enum SharpAIPanguUltraMoeSanitizer {

    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIPanguUltraMoeConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
                && !isMultiTokenPredictionKey(key, hiddenLayers: configuration.hiddenLayers)
        }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        return sanitized
    }

    /// True for keys under `model.layers.<i>` with i at or past the main stack.
    static func isMultiTokenPredictionKey(_ key: String, hiddenLayers: Int) -> Bool {
        let parts = key.split(separator: ".")
        guard parts.count >= 3, parts[0] == "model", parts[1] == "layers",
            let layer = Int(parts[2])
        else {
            return false
        }
        return layer >= hiddenLayers
    }
}
