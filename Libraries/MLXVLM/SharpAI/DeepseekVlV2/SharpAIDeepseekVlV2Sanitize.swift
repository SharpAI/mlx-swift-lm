// SharpAI fork overlay: deepseek_vl_v2 checkpoint key sanitization. Not part of upstream.
//
// Verified against: https://huggingface.co/deepseek-ai/deepseek-vl2-tiny/blob/main/model.safetensors.index.json
// (weight_map keys: language.model.*, language.lm_head.*, vision.*, projector.layers.*,
// image_newline, view_seperator).
// Text keys keep their HF names so DeepseekV2Model.sanitize can run on the output.

import Foundation
import MLX

public enum SharpAIDeepseekVlV2Sanitizer {

    static let languagePrefix = "language."
    static let visionPrefixes = [
        "vision.", "projector.", "sam_model.", "vision_model.",
        "model.sam_model.", "model.vision_model.", "model.projector.",
    ]
    static let visionKeys: Set<String> = [
        "image_newline", "view_seperator", "model.image_newline", "model.view_seperator",
    ]

    /// Drops vision-side weights and strips the language prefix from text weights.
    public static func sanitize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if visionKeys.contains(key) || visionPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }

            var newKey = key
            if newKey.hasPrefix(languagePrefix) {
                newKey = String(newKey.dropFirst(languagePrefix.count))
            }

            guard sanitized[newKey] == nil else {
                throw SharpAIDeepseekVlV2Error.duplicateWeightKey(newKey)
            }
            sanitized[newKey] = value
        }

        return sanitized
    }
}
