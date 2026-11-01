import Foundation
import MLX

/// Maps checkpoint keys onto `SharpAIQwen3VlNemotronEmbedModel` module paths.
/// Verified against: https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/resolve/main/model.safetensors.index.json
/// (keys are `model.visual.*`, `model.language_model.*` and `lm_head.weight`; there is no embedding head key).
public enum SharpAIQwen3VlNemotronEmbedSanitize {

    public static func sanitize(
        weights: [String: MLXArray], config: SharpAIQwen3VlNemotronEmbedConfiguration
    ) -> [String: MLXArray] {
        let tied = config.backbone.textConfiguration.tieWordEmbeddings
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)

        for (key, value) in weights {
            var name = key

            // Qwen3-VL style: model.visual -> visual, model.language_model -> language_model.
            if name.hasPrefix("model.") {
                name = String(name.dropFirst("model.".count))
            }

            if name.hasPrefix("visual.") {
                name = "vision_tower." + String(name.dropFirst("visual.".count))
            }

            if name.hasPrefix("lm_head.") {
                if tied { continue }
                name = "language_model." + name
            }

            if name.hasPrefix("language_model.lm_head.") {
                if tied { continue }
            } else if name.hasPrefix("language_model.")
                && !name.hasPrefix("language_model.model.")
            {
                name = "language_model.model." + String(name.dropFirst("language_model.".count))
            }

            adjusted[name] = value
        }

        // Drops position_ids and transposes the patch embedding, as in Qwen3VL.
        return Qwen3VLVision.VisionModel.sanitize(
            weights: adjusted, config: config.backbone.visionConfiguration)
    }
}
