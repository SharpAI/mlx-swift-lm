// SharpAI fork overlay: muse_glimmer_text checkpoint key mapping. Not part of upstream
// mlx-swift-lm.
//
// The text model's module tree is `model.*` and `lm_head.*`, the layout of a text-only
// checkpoint. Other layouts seen for this family map onto it. Vision weights are dropped,
// since a text-only model has no tower for them.

import Foundation
import MLX

public enum SharpAIMuseGlimmerTextSanitize {

    /// Maps checkpoint keys to module keys and drops keys this model does not load.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray](minimumCapacity: weights.count)
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") { continue }
            if isVisionKey(key) { continue }
            result[remapKey(key)] = value
        }
        return result
    }

    /// `model.language_model.X` (VLM layout), `language_model.model.X` and
    /// `language_model.lm_head.X` map to `model.X` and `lm_head.X`. Other keys pass through.
    static func remapKey(_ key: String) -> String {
        if key.hasPrefix("model.language_model.") {
            return "model." + String(key.dropFirst("model.language_model.".count))
        }
        if key.hasPrefix("language_model.model.") {
            return "model." + String(key.dropFirst("language_model.model.".count))
        }
        if key.hasPrefix("language_model.lm_head.") {
            return "lm_head." + String(key.dropFirst("language_model.lm_head.".count))
        }
        return key
    }

    /// True for vision-tower, vision-adapter, projector and perception keys in either layout.
    static func isVisionKey(_ key: String) -> Bool {
        var stripped = key
        if stripped.hasPrefix("model.") {
            stripped = String(stripped.dropFirst("model.".count))
        }
        return stripped.hasPrefix("vision_") || stripped.hasPrefix("perception_emb_norm.")
    }
}
