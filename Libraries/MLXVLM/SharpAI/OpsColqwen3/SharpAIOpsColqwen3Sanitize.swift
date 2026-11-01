import Foundation
import MLX

/// Maps ops_colqwen3 checkpoint keys onto `SharpAIOpsColqwen3Model` module paths.
///
/// UNVERIFIED: the key layout is inferred from Qwen3-VL and ColQwen conventions. No
/// OpenSearch checkpoint is on disk to confirm it.
public enum SharpAIOpsColqwen3Sanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIOpsColqwen3Configuration
    ) -> [String: MLXArray] {
        let tied = config.backbone.textConfiguration.tieWordEmbeddings
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var name = key

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
