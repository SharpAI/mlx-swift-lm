// SharpAI fork overlay: nemotron_h_omni checkpoint key sanitization. Not part of upstream.
//
// Text keys under `language_model.` lose that prefix, so NemotronHModel.sanitize sees the
// `backbone.` and `lm_head.` names it expects. Tower weights are dropped.
// Verified against: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/model.safetensors.index.json
// The weight_map keys seen there are `language_model.backbone.layers...`, `vision_model.radio_model...`,
// `mlp1.3.weight` and `sound_encoder...`. `sound_projection` was not visible in the read, so it is
// covered only by the `sound_` prefix, not by a seen key.

import Foundation
import MLX

public enum SharpAINemotronHOmniSanitizer {

    public static let languagePrefix = "language_model."
    public static let towerPrefixes = ["vision_model.", "vision_tower.", "mlp1.", "sound_"]

    /// Drops tower weights and strips `language_model.` from text weights.
    public static func textWeights(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var text = [String: MLXArray]()

        for (key, value) in weights {
            if towerPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var textKey = key
            if textKey.hasPrefix(languagePrefix) {
                textKey = String(textKey.dropFirst(languagePrefix.count))
            }

            guard text[textKey] == nil else {
                throw SharpAINemotronHOmniError.duplicateWeightKey(textKey)
            }
            text[textKey] = value
        }

        return text
    }
}
