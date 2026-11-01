// SharpAI fork overlay: ColPali checkpoint key remap. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Maps ColPali checkpoint keys onto `SharpAIColpali` module paths.
public enum SharpAIColpaliSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        renamed.reserveCapacity(weights.count)
        for (key, value) in weights {
            guard let name = mappedName(key) else { continue }
            renamed[name] = value
        }
        return SharpAIColpaliVision.VisionTower.sanitize(weights: renamed)
    }

    /// Module path for one checkpoint key, or nil to drop it.
    ///
    /// UNVERIFIED: the checkpoint prefixes are not checked against a real ColPali file.
    /// Handled: `vlm.` / `model.` wrappers, `custom_text_proj` (colpali-engine name) for
    /// `embedding_proj_layer`, and a missing `model.` inside `language_model.`.
    static func mappedName(_ key: String) -> String? {
        // Embeddings are tied, so the LM head is unused. Rotary caches are recomputed.
        if key.hasSuffix("lm_head.weight") || key.contains("rotary_emb.inv_freq") {
            return nil
        }

        var body = key
        if body.hasPrefix("vlm.") {
            body = String(body.dropFirst("vlm.".count))
        }
        if body.hasPrefix("model.") {
            body = String(body.dropFirst("model.".count))
        }

        if body.hasPrefix("custom_text_proj.") {
            return "embedding_proj_layer." + String(body.dropFirst("custom_text_proj.".count))
        }
        if body.hasPrefix("embedding_proj_layer.") {
            return body
        }

        let backboneRoots = ["vision_tower.", "language_model.", "multi_modal_projector."]
        guard backboneRoots.contains(where: { body.hasPrefix($0) }) else {
            return key
        }

        var path = "vlm." + body
        let languagePrefix = "vlm.language_model."
        if path.hasPrefix(languagePrefix), !path.hasPrefix(languagePrefix + "model.") {
            path = languagePrefix + "model." + String(path.dropFirst(languagePrefix.count))
        }
        return path
    }
}
