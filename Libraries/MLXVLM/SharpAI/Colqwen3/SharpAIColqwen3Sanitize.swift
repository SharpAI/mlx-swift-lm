import Foundation
import MLX

/// Maps ColQwen3 checkpoint keys onto `SharpAIColqwen3Model` module paths.
public enum SharpAIColqwen3Sanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIColqwen3Configuration
    ) -> [String: MLXArray] {
        let tied = config.backbone.textConfiguration.tieWordEmbeddings
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var name = key

            // UNVERIFIED: a leading "vlm." wrapper prefix (colqwen2-style). The tomoro
            // colqwen3 remote code is not on disk, so this is not checked against it.
            if name.hasPrefix("vlm.") {
                name = String(name.dropFirst("vlm.".count))
            }

            // Qwen3-VL style "model." prefix: model.visual -> visual, model.language_model -> language_model.
            if name.hasPrefix("model.") {
                name = String(name.dropFirst("model.".count))
            }

            if name.hasPrefix("visual.") {
                name = "vision_tower." + String(name.dropFirst("visual.".count))
            }

            if name.hasPrefix("lm_head.") {
                name = "language_model." + name
            }

            if name.hasPrefix("language_model.lm_head.") {
                // Tied embeddings have no separate head; the logits head is unused here.
                if tied { continue }
            } else if name.hasPrefix("language_model.") && !name.hasPrefix("language_model.model.") {
                name = "language_model.model." + String(name.dropFirst("language_model.".count))
            }

            adjusted[name] = value
        }

        // Drops position_ids and transposes the patch embedding, as in Qwen3VL.
        return Qwen3VLVision.VisionModel.sanitize(
            weights: adjusted, config: config.backbone.visionConfiguration)
    }
}
