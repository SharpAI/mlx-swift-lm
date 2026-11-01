// SharpAI fork overlay: ModernVBERT model and ColModernVBert retrieval head.
// Not part of upstream mlx-swift-lm.
//
// `embed` returns one vector per token. With a retrieval head, the vectors are unit norm.
// Without one (plain ModernVBertModel), it returns the encoder hidden states. The
// LanguageModel conformance exists only so the model fits the creator type. Its "logits"
// are these vectors, so do not sample from it. Not implemented: the ColModernVBert
// processor (query prefix and pad augmentation).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `model.*` in the checkpoint: `vision_model`, `connector` and `text_model`.
final class SharpAIModernvbertBackbone: Module {
    @ModuleInfo(key: "vision_model") var visionModel: SharpAIModernvbertVision.Transformer
    @ModuleInfo var connector: SharpAIModernvbertVision.Connector
    @ModuleInfo(key: "text_model") var textModel: SharpAIModernvbertText.TextModel

    init(_ config: SharpAIModernvbertConfiguration) {
        self._visionModel.wrappedValue = SharpAIModernvbertVision.Transformer(
            config.visionConfiguration)
        self._connector.wrappedValue = SharpAIModernvbertVision.Connector(
            config.visionConfiguration, factor: config.pixelShuffleFactor,
            textHidden: config.textConfiguration.hiddenSize)
        self._textModel.wrappedValue = SharpAIModernvbertText.TextModel(config.textConfiguration)
    }
}

/// ModernVBERT encoder with an optional ColModernVBert projection (`embedding_proj_layer`).
public final class SharpAIModernvbert: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "model") var model: SharpAIModernvbertBackbone
    /// Nil for plain `ModernVBertModel` checkpoints.
    // UNVERIFIED: bias. The reference projection is assumed to have none.
    @ModuleInfo(key: "embedding_proj_layer") var embeddingProjLayer: Linear?

    public let config: SharpAIModernvbertConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.attentionHeads,
            count: config.textConfiguration.hiddenLayers)
    }
    public var loraLayers: [Module] { model.textModel.layers }

    public init(_ config: SharpAIModernvbertConfiguration) {
        self.config = config
        self._model.wrappedValue = SharpAIModernvbertBackbone(config)
        if config.hasRetrievalHead {
            self._embeddingProjLayer.wrappedValue = Linear(
                config.textConfiguration.hiddenSize, config.embeddingDimensions, bias: false)
        } else {
            self._embeddingProjLayer.wrappedValue = nil
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIModernvbertSanitize.sanitize(weights: weights)
    }

    /// Token vectors `[batch, seq, embeddingDimensions]`, or hidden states without a head.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` token ids.
    ///   - pixelValues: `[batch, channels, height, width]` NCHW pixels, if any. Each image
    ///     needs exactly `config.imageTokenCount` image tokens in `inputIds`.
    ///   - attentionMask: `[batch, seq]`, nonzero for real tokens. Nil means no padding.
    public func embed(
        _ inputIds: MLXArray, pixelValues: MLXArray? = nil, attentionMask: MLXArray? = nil
    ) throws -> MLXArray {
        let hidden = try encode(inputIds, pixelValues: pixelValues, attentionMask: attentionMask)
        guard let projection = embeddingProjLayer else {
            return hidden
        }
        let vectors = l2Normalize(projection(hidden))
        guard let attentionMask else {
            return vectors
        }
        return vectors * attentionMask.asType(vectors.dtype).expandedDimensions(axis: -1)
    }

    /// Text-only forward. Returns the embeddings in the logits slot.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // Text-only input cannot throw; only image merging throws.
        try! embed(inputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let embeddings = try embed(
            tokens, pixelValues: input.image?.pixels, attentionMask: input.text.mask)
        let total = tokens.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: embeddings))
    }

    private func encode(
        _ inputIds: MLXArray, pixelValues: MLXArray?, attentionMask: MLXArray?
    ) throws -> MLXArray {
        let text = model.textModel
        var tokenEmbeds = text.embeddings.tokEmbeddings(inputIds)
        if let pixelValues {
            let features = imageFeatures(pixelValues, dtype: tokenEmbeds.dtype)
            tokenEmbeds = try mergeImageFeatures(features, into: tokenEmbeds, inputIds: inputIds)
        }
        return text.encode(tokenEmbeds, paddingMask: attentionMask.map { $0.asType(.bool) })
    }

    /// Connector output, `[batch, image tokens, text hidden]`.
    private func imageFeatures(_ pixelValues: MLXArray, dtype: DType) -> MLXArray {
        let visual = model.visionModel(pixelValues.transposed(0, 2, 3, 1).asType(dtype))
        return model.connector(visual)
    }

    /// Writes image features into the image-token rows, in order.
    private func mergeImageFeatures(
        _ features: MLXArray, into textEmbeds: MLXArray, inputIds: MLXArray
    ) throws -> MLXArray {
        let hiddenSize = textEmbeds.dim(-1)
        let imageFlags = (inputIds .== MLXArray(config.imageTokenIndex)).asArray(Bool.self)
        let indices = imageFlags.enumerated().filter { $0.element }.map { Int32($0.offset) }

        let featureRows = features.dim(0) * features.dim(1)
        guard indices.count == featureRows else {
            throw SharpAIModernvbertError.imageTokenMismatch(
                expected: featureRows, actual: indices.count)
        }
        guard !indices.isEmpty else { return textEmbeds }

        let flat = textEmbeds.reshaped([-1, hiddenSize])
        flat[MLXArray(indices), 0...] = features.reshaped([featureRows, hiddenSize])
        return flat.reshaped(textEmbeds.shape)
    }

    private func l2Normalize(_ x: MLXArray) -> MLXArray {
        x / sqrt((x * x).sum(axis: -1, keepDims: true))
    }
}
