// SharpAI fork overlay: ColPali retrieval model. Not part of upstream mlx-swift-lm.
//
// ColPali is a late-interaction retriever, not a generator. `embed` returns one unit-norm
// vector per token. The LanguageModel conformance exists only so the model fits the
// creator type. Its "logits" are these embeddings, so do not sample from it.
// Not implemented: the ColPali processor (query prefix and pad augmentation).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Linear map from SigLIP features to the text width. Module path `multi_modal_projector.*`.
final class SharpAIColpaliProjector: Module {
    @ModuleInfo var linear: Linear

    init(_ config: SharpAIColpaliConfiguration.VisionConfiguration) {
        self.linear = Linear(config.hiddenSize, config.projectionDimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }
}

/// PaliGemma layout under `vlm.`: `vision_tower`, `language_model`, `multi_modal_projector`.
final class SharpAIColpaliBackbone: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIColpaliVision.VisionTower
    @ModuleInfo(key: "language_model") var languageModel: SharpAIColpaliText.TextModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: SharpAIColpaliProjector

    init(_ config: SharpAIColpaliConfiguration) {
        self._visionTower.wrappedValue = SharpAIColpaliVision.VisionTower(config.visionConfiguration)
        self._languageModel.wrappedValue = SharpAIColpaliText.TextModel(config.textConfiguration)
        self._multiModalProjector.wrappedValue = SharpAIColpaliProjector(config.visionConfiguration)
    }
}

/// ColPali retrieval encoder: PaliGemma backbone, linear projection to `embedding_dim`,
/// L2 normalization per token. Weight keys: `vlm.*` and `embedding_proj_layer.*`.
public final class SharpAIColpali: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vlm") var vlm: SharpAIColpaliBackbone
    // UNVERIFIED: bias. The reference projection is assumed to have one.
    @ModuleInfo(key: "embedding_proj_layer") var embeddingProjLayer: Linear

    public let config: SharpAIColpaliConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { vlm.languageModel.kvHeads }
    public var loraLayers: [Module] { vlm.languageModel.model.layers }

    public init(_ config: SharpAIColpaliConfiguration) {
        self.config = config
        self._vlm.wrappedValue = SharpAIColpaliBackbone(config)
        self._embeddingProjLayer.wrappedValue = Linear(
            config.textConfiguration.hiddenSize, config.embeddingDimensions, bias: true)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIColpaliSanitize.sanitize(weights: weights)
    }

    /// Unit-norm token vectors `[batch, seq, embeddingDimensions]`.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` token ids.
    ///   - pixelValues: `[batch, channels, height, width]` NCHW pixels, if any. Each image
    ///     needs exactly `(image_size / patch_size)^2` image tokens in `inputIds`.
    public func embed(
        _ inputIds: MLXArray, pixelValues: MLXArray? = nil, cache: [any KVCache]? = nil
    ) throws -> MLXArray {
        let textModel = vlm.languageModel
        let textEmbeds = textModel.model.embedTokens(inputIds)

        let hidden: MLXArray
        if let pixelValues {
            let features = imageFeatures(pixelValues, dtype: textEmbeds.dtype)
            let merged = try mergeImageFeatures(features, into: textEmbeds, inputIds: inputIds)
            hidden = textModel.model(
                inputIds, cache: cache, inputEmbedding: merged, bidirectional: true)
        } else {
            hidden = textModel.model(
                inputIds, cache: cache, inputEmbedding: textEmbeds, bidirectional: false)
        }

        return l2Normalize(embeddingProjLayer(hidden))
    }

    /// Text-only forward. Returns the embeddings in the logits slot.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // Text-only input cannot throw; only image merging throws.
        try! embed(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let embeddings = try embed(tokens, pixelValues: input.image?.pixels, cache: cache)
        let total = tokens.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: embeddings))
    }

    /// SigLIP features scaled by `1 / sqrt(text hidden)`, as in PaliGemma. Shape `[batch, patches, text hidden]`.
    private func imageFeatures(_ pixelValues: MLXArray, dtype: DType) -> MLXArray {
        let visual = vlm.visionTower(pixelValues.transposed(0, 2, 3, 1).asType(dtype))
        let projected = vlm.multiModalProjector(visual)
        return projected / pow(Float(config.textConfiguration.hiddenSize), 0.5)
    }

    /// Writes image features into the image-token rows, in order. Zeroes pad rows.
    private func mergeImageFeatures(
        _ features: MLXArray, into textEmbeds: MLXArray, inputIds: MLXArray
    ) throws -> MLXArray {
        let hiddenSize = textEmbeds.dim(-1)
        let imageFlags = (inputIds .== MLXArray(config.imageTokenIndex)).asArray(Bool.self)
        let indices = imageFlags.enumerated().filter { $0.element }.map { Int32($0.offset) }

        let featureRows = features.dim(0) * features.dim(1)
        guard indices.count == featureRows else {
            throw SharpAIColpaliError.imageTokenMismatch(
                expected: featureRows, actual: indices.count)
        }
        guard !indices.isEmpty else { return textEmbeds }

        let flat = textEmbeds.reshaped([-1, hiddenSize])
        flat[MLXArray(indices), 0...] = features.reshaped([featureRows, hiddenSize])

        let padMask = (inputIds .== MLXArray(config.padTokenId)).expandedDimensions(axis: -1)
        return which(padMask, 0, flat.reshaped(textEmbeds.shape))
    }

    private func l2Normalize(_ x: MLXArray) -> MLXArray {
        x / sqrt((x * x).sum(axis: -1, keepDims: true))
    }
}
