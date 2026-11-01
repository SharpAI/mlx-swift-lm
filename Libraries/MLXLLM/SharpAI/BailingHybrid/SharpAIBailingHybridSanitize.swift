// SharpAI fork overlay: bailing_hybrid checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keeps HF key names. Only drops keys the Swift model does not load.
// UNVERIFIED: MTP/nextn layers are assumed to sit at indices num_hidden_layers..<num_hidden_layers + num_nextn_predict_layers.
// Keys named differently are kept, not dropped.
// Expert stacking is not done here (same as BailingMoe.swift). Confirm both against a real checkpoint.

import Foundation
import MLX

public enum SharpAIBailingHybridSanitizer {

    /// Drops tied `lm_head` weights and MTP/nextn layer weights. Other keys keep their HF names.
    public static func sanitize(
        _ weights: [String: MLXArray], configuration: SharpAIBailingHybridConfiguration
    ) -> [String: MLXArray] {
        weights.filter { key, _ in
            if configuration.tieWordEmbeddings,
                key.split(separator: ".").contains("lm_head")
            {
                return false
            }
            if let index = layerIndex(of: key),
                index >= configuration.hiddenLayers,
                index < configuration.hiddenLayers + configuration.numNextnPredictLayers
            {
                return false
            }
            return true
        }
    }

    /// Returns N for keys shaped like `...layers.N.<rest>`, or nil.
    public static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let at = parts.firstIndex(of: "layers"), at + 1 < parts.count else {
            return nil
        }
        return Int(parts[at + 1])
    }
}
