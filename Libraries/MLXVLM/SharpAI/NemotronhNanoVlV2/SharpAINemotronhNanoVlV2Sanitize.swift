// SharpAI fork overlay: NemotronH_Nano_VL_V2 checkpoint key sanitization. Not part of upstream.
//
// Verified against: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16/resolve/main/model.safetensors.index.json
// The checkpoint uses `vision_model.radio_model.*`, `mlp1.{0,1,3}.*`, `language_model.backbone.*`
// and `language_model.lm_head.*`. The `vision_tower.` prefix is not in that checkpoint; it is
// kept only as a harmless extra drop rule.
// Text keys keep their HF names so NemotronHModel.sanitize can run on the output.

import Foundation
import MLX

public enum SharpAINemotronhNanoVlV2Sanitizer {

    static let languagePrefix = "language_model."
    static let visionPrefixes = ["vision_model.", "vision_tower.", "mlp1."]

    /// Drops vision-side weights and strips the language prefix from text weights.
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
                throw SharpAINemotronhNanoVlV2Error.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
