// SharpAI fork overlay: qwen3_5_moe_text model. Not part of upstream mlx-swift-lm.
//
// Subclasses the upstream text model, so the forward pass, cache, MTP and LoRA behaviour are the
// upstream ones. The only change is the expert stacking step in `sanitize`. Both classes are in
// the MLXLLM module, so the subclass is allowed without touching the upstream file.

import Foundation
import MLX

public class SharpAIQwen35MoeTextModel: Qwen35TextModel {

    override public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let stacked = SharpAIQwen35MoeTextExperts.stack(
            weights,
            layers: configuration.hiddenLayers,
            numExperts: configuration.numExperts,
            mtpLayers: configuration.numNextnPredictLayers)
        return super.sanitize(weights: stacked)
    }
}
