// SharpAI fork overlay: StepVL vision-language model. Not part of upstream mlx-swift-lm.
//
// Perception Encoder features go through `vit_large_projector` into the text width. Each image
// placeholder token (`image_token_id`) in the prompt takes the next projected feature row, in
// order. There is no image processor here, so callers pass normalized NCHW pixels with 3 channels.
// UNVERIFIED: the placeholder layout and any start/end tokens around image features are not
// checked against a real StepVL checkpoint. vLLM notes that it does not insert them either.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// StepVL (`model_type` "step_robotics"). Weight keys follow `SharpAIStepRoboticsSanitize`:
/// `vision_model.*`, `vit_large_projector.*`, `language_model.model.*`, `language_model.lm_head`.
public final class SharpAIStepRoboticsModel: Module, VLMModel, KVCacheDimensionProvider {

    public let config: SharpAIStepRoboticsConfiguration

    @ModuleInfo(key: "vision_model") var visionModel: SharpAIStepRoboticsVisionEncoder
    @ModuleInfo(key: "vit_large_projector") var vitLargeProjector: Linear
    @ModuleInfo(key: "language_model") var languageModel: SharpAIStepRoboticsLanguageModel

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: SharpAIStepRoboticsConfiguration) throws {
        self.config = config
        self._visionModel.wrappedValue = try SharpAIStepRoboticsVisionEncoder(
            config.visionConfiguration)
        // The encoder emits `width * 4` channels after its two downsampling convs.
        self._vitLargeProjector.wrappedValue = Linear(
            config.visionConfiguration.width * 4, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
        self._languageModel.wrappedValue = SharpAIStepRoboticsLanguageModel(
            config.textConfiguration)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIStepRoboticsSanitize.sanitize(weights: weights)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let logits: MLXArray
        if let pixels = input.image?.pixels {
            let embeddings = try multimodalEmbeddings(tokens, pixelValues: pixels)
            logits = languageModel.logits(
                languageModel.model(embeddings: embeddings, cache: cache))
        } else {
            logits = languageModel(tokens, cache: cache)
        }

        let total = tokens.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// NCHW pixels `[batch, 3, H, W]` to projected tokens `[batch * images, tokens, text hidden]`.
    func imageFeatures(_ pixelValues: MLXArray, dtype: DType) throws -> MLXArray {
        guard pixelValues.ndim == 4 else {
            throw SharpAIStepRoboticsError.imageChannelMismatch(pixelValues.ndim)
        }
        guard pixelValues.dim(1) == 3 else {
            throw SharpAIStepRoboticsError.imageChannelMismatch(pixelValues.dim(1))
        }
        let nhwc = pixelValues.transposed(0, 2, 3, 1).asType(dtype)
        let visual = try visionModel(nhwc)
        return vitLargeProjector(visual)
    }

    /// Text embeddings with image features written into the placeholder slots, in order.
    /// The features must match the placeholder count exactly.
    func multimodalEmbeddings(_ tokens: MLXArray, pixelValues: MLXArray) throws -> MLXArray {
        let textEmbeds = languageModel.embed(tokens)
        let hidden = config.textConfiguration.hiddenSize
        let features = try imageFeatures(pixelValues, dtype: textEmbeds.dtype)
        let flatFeatures = features.reshaped(-1, hidden)

        let placeholderMask = (tokens .== MLXArray(Int32(config.imageTokenId))).reshaped(-1)
        let placeholderCount = Int(
            placeholderMask.asType(.int32).sum().asArray(Int32.self).first ?? 0)
        guard placeholderCount == flatFeatures.dim(0) else {
            throw SharpAIStepRoboticsError.placeholderMismatch(
                placeholders: placeholderCount, features: flatFeatures.dim(0))
        }

        // Slot i of the placeholders takes feature row i.
        let cursor = cumsum(placeholderMask.asType(.int32), axis: 0) - MLXArray(Int32(1))
        let rows = flatFeatures.take(maximum(cursor, MLXArray(Int32(0))), axis: 0)
        let flatText = textEmbeds.reshaped(-1, hidden)
        let merged = `where`(placeholderMask.expandedDimensions(axis: -1), rows, flatText)
        return merged.reshaped(tokens.dim(0), tokens.dim(1), hidden)
    }
}
