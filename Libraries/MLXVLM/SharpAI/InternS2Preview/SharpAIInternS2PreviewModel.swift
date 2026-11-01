// SharpAI fork overlay: intern_s2_preview model. Not part of upstream mlx-swift-lm.
//
// vLLM aliases this architecture to Qwen3_5MoeForConditionalGeneration, so the forward pass is
// the upstream Qwen35. This subclass adds the time-series drop and MoE fusion to sanitize.
// Qwen35MoE is final, so its fusion is repeated via SharpAIInternS2PreviewSanitizer.

import Foundation
import MLX
import MLXLMCommon

public final class SharpAIInternS2PreviewModel: Qwen35 {

    public override init(_ config: Qwen35Configuration) {
        super.init(config)
    }

    public override func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        let filtered = SharpAIInternS2PreviewSanitizer.dropTimeSeries(weights)
        let fused = SharpAIInternS2PreviewSanitizer.fuseExperts(
            filtered, hiddenLayers: config.textConfiguration.hiddenLayers)
        return try super.sanitize(weights: fused)
    }

    // The real load path calls prepareCheckpoint, whose prepareTarget step renames `model.*` to
    // `language_model.model.*` before sanitize runs, so the drop has to happen before it.
    public override func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        var checkpoint = checkpoint
        checkpoint.weights = SharpAIInternS2PreviewSanitizer.dropTimeSeries(checkpoint.weights)
        return try super.prepareCheckpoint(checkpoint)
    }

    // Drops time-series keys before Qwen35 remaps bare `model.*` keys, so they never reach the
    // language model. Converted MLX checkpoints take the metadata path and skip fusion, as in Qwen35.
    public override func sanitize(
        weights: [String: MLXArray], metadata: [String: String]
    ) throws -> [String: MLXArray] {
        try super.sanitize(
            weights: SharpAIInternS2PreviewSanitizer.dropTimeSeries(weights),
            metadata: metadata)
    }
}
