// SharpAI fork overlay: llama_nemotron_vl_rerank forward pass. Not part of upstream mlx-swift-lm.
//
// Mirrors LlamaNemotronVLForSequenceClassification in modeling_llama_nemotron_vl.py:
//   SigLIP tower -> pixel shuffle (always the v2 permute) -> mlp1 -> image rows merged into the
//   text embeddings -> bidirectional Llama (no causal mask) -> average over tokens ->
//   score head (no bias) -> divide by temperature.
// The vision and text stacks are the sibling LlamaNemotronVl embedder's layers, reused as they are.
//
// Limits:
//   - Batches must not be padded. The reference masks padded tokens in attention and in pooling.
//     The sibling decoder takes no mask, so this path does not mask them.
//   - Pixels must already be preprocessed (SigLIP mean 0.5 and std 0.5). The dynamic tiling and
//     thumbnail logic of the reference processor is not implemented here.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Reranker that returns one score per sequence. The `LanguageModel` conformance exists only so the
/// model fits the creator type. Its "logits" are these scores, so do not sample from it.
public final class SharpAILlamaNemotronVlRerankModel: Module, VLMModel, KVCacheDimensionProvider {

    /// Module path `model.vision_model.vision_model.*`. The reference SiglipVisionModel adds one level.
    final class VisionTower: Module {
        @ModuleInfo(key: "vision_model") var transformer: SharpAILlamaNemotronVlVision.Transformer

        init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
            self._transformer.wrappedValue = SharpAILlamaNemotronVlVision.Transformer(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            transformer(x)
        }
    }

    /// Module path `model.*`.
    final class Backbone: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionTower
        @ModuleInfo(key: "mlp1") var mlp1: SharpAILlamaNemotronVlProjector.MLP1
        @ModuleInfo(key: "language_model") var languageModel: SharpAILlamaNemotronVlLanguage.Model

        init(
            text: SharpAILlamaNemotronVlConfiguration.TextConfiguration,
            vision: SharpAILlamaNemotronVlConfiguration.VisionConfiguration,
            factor: Int
        ) {
            self._visionModel.wrappedValue = VisionTower(vision)
            self._mlp1.wrappedValue = SharpAILlamaNemotronVlProjector.MLP1(
                inputDimensions: vision.hiddenSize * factor * factor,
                hiddenDimensions: text.hiddenSize,
                outputDimensions: text.hiddenSize)
            self._languageModel.wrappedValue = SharpAILlamaNemotronVlLanguage.Model(text)
        }
    }

    public let config: SharpAILlamaNemotronVlRerankConfiguration
    private let imageSize: Int
    private let factor: Int

    @ModuleInfo(key: "model") var backbone: Backbone
    /// Module path `score.*`. Single label, no bias, float32 in the reference.
    @ModuleInfo(key: "score") var scoreHead: Linear

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { backbone.languageModel.kvHeads }
    public var loraLayers: [Module] { backbone.languageModel.layers }

    /// Throws when the config has no SigLIP dimensions, since the model cannot be built without them.
    public init(_ config: SharpAILlamaNemotronVlRerankConfiguration) throws {
        guard let vision = config.visionConfiguration else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: vision_config with SigLIP dimensions is required to build the model")
        }
        self.config = config
        self.imageSize = vision.imageSize
        self.factor = config.pixelShuffleFactor
        let text = config.textConfiguration

        self._backbone.wrappedValue = Backbone(text: text, vision: vision, factor: factor)
        self._scoreHead.wrappedValue = Linear(text.hiddenSize, 1, bias: false)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAILlamaNemotronVlRerankSanitize.sanitize(weights)
    }

    /// One relevance score per sequence, shape `[batch, 1]`.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` token ids. Unpadded.
    ///   - pixelValues: `[images, channels, height, width]` NCHW pixels, if any. Each image needs
    ///     exactly `(image_size / patch_size / k)^2` rows of `img_context_token_id` in `inputIds`, in order.
    public func rerankScores(_ inputIds: MLXArray, pixelValues: MLXArray? = nil) throws -> MLXArray {
        let hidden = try hiddenStates(inputIds, pixelValues: pixelValues)
        // avg pooling: mean over the token axis, as `pool(..., "avg")` with an all-ones mask.
        let pooled = hidden.mean(axis: 1).asType(.float32)
        return scoreHead(pooled) / config.temperature
    }

    /// Hidden states `[batch, seq, text hidden]` after the final norm.
    func hiddenStates(_ inputIds: MLXArray, pixelValues: MLXArray?) throws -> MLXArray {
        let textEmbeds = backbone.languageModel.embedTokens(inputIds)
        let merged: MLXArray
        if let pixelValues {
            let features = try imageFeatures(pixelValues, dtype: textEmbeds.dtype)
            merged = try mergeImageFeatures(features, into: textEmbeds, inputIds: inputIds)
        } else {
            merged = textEmbeds
        }
        return backbone.languageModel(inputEmbeddings: merged)
    }

    /// Vision features as `[images * tokens, text hidden]`, after pixel shuffle and projection.
    private func imageFeatures(_ pixelValues: MLXArray, dtype: DType) throws -> MLXArray {
        guard pixelValues.dim(2) == imageSize, pixelValues.dim(3) == imageSize else {
            throw SharpAILlamaNemotronVlRerankError.invalidPixelShape(
                expected: imageSize, actual: pixelValues.dim(2))
        }
        let nhwc = pixelValues.transposed(0, 2, 3, 1).asType(dtype)
        let patches = backbone.visionModel(nhwc)

        let count = patches.dim(1)
        let side = Int(Double(count).squareRoot())
        let grid = patches.reshaped(patches.dim(0), side, side, patches.dim(2))
        // The reference pixel_shuffle always does the second transpose, so ps_version is not read here.
        let shuffled = SharpAILlamaNemotronVlProjector.pixelShuffle(
            grid, factor: factor, version: "v2")
        let tokens = shuffled.reshaped(shuffled.dim(0), -1, shuffled.dim(3))
        let projected = backbone.mlp1(tokens)
        return projected.reshaped(-1, projected.dim(-1))
    }

    /// Writes image features into the image-token rows, in order.
    private func mergeImageFeatures(
        _ features: MLXArray, into textEmbeds: MLXArray, inputIds: MLXArray
    ) throws -> MLXArray {
        guard let imageTokenId = config.imageTokenId else {
            throw SharpAILlamaNemotronVlRerankError.missingImageTokenId
        }
        let hiddenSize = textEmbeds.dim(-1)
        let imageFlags = (inputIds .== MLXArray(Int32(imageTokenId))).asArray(Bool.self)
        let indices = imageFlags.enumerated().filter { $0.element }.map { Int32($0.offset) }

        let featureRows = features.dim(0)
        guard indices.count == featureRows else {
            throw SharpAILlamaNemotronVlRerankError.imageTokenMismatch(
                expected: featureRows, actual: indices.count)
        }
        guard !indices.isEmpty else { return textEmbeds }

        let flat = textEmbeds.reshaped([-1, hiddenSize])
        flat[MLXArray(indices), 0...] = features.reshaped([featureRows, hiddenSize])
        return flat.reshaped(textEmbeds.shape)
    }

    /// Text-only forward. Returns the scores in the logits slot. This path cannot throw.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        try! rerankScores(inputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let scores = try rerankScores(tokens, pixelValues: input.image?.pixels)
        let total = tokens.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: scores))
    }
}
