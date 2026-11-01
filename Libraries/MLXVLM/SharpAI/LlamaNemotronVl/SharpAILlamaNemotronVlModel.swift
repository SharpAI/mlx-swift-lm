// SharpAI fork overlay: LlamaNemotronVL embedder. Not part of upstream mlx-swift-lm.
//
// SigLIP vision, pixel shuffle, MLP projector into the text width, then a bidirectional Llama
// decoder. `embed` returns per-token hidden states. The LanguageModel conformance exists only
// so the model fits the creator type. Its "logits" are these hidden states, so do not sample from it.
//
// Not implemented: the C-RADIO chat variant, the reranker head, and the image processor. The
// caller must pass pixels already normalized with mean 0.5 and std 0.5 per channel (SigLIP).
// UNVERIFIED: the pooling used for the final embedding. `meanPooled` is a guess, not the reference.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Wraps the SigLIP transformer one level deeper, matching the checkpoint's
/// `vision_model.vision_model.*` paths (the reference `SiglipVisionModel` adds that level).
final class SharpAILlamaNemotronVlVisionTower: Module {
    @ModuleInfo(key: "vision_model") var transformer: SharpAILlamaNemotronVlVision.Transformer

    init(_ config: SharpAILlamaNemotronVlConfiguration.VisionConfiguration) {
        self._transformer.wrappedValue = SharpAILlamaNemotronVlVision.Transformer(config)
    }
}

/// VLM embedder with a SigLIP tower, a pixel-shuffle projector, and a bidirectional Llama decoder.
/// Weight keys: `vision_model.vision_model.*`, `mlp1.*`, `language_model.*`.
public final class SharpAILlamaNemotronVl: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_model") var visionModel: SharpAILlamaNemotronVlVisionTower
    @ModuleInfo var mlp1: SharpAILlamaNemotronVlProjector.MLP1
    @ModuleInfo(key: "language_model") var languageModel: SharpAILlamaNemotronVlLanguage.Model

    public let config: SharpAILlamaNemotronVlConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.layers }

    public init(_ config: SharpAILlamaNemotronVlConfiguration) {
        self.config = config
        let text = config.textConfiguration
        let vision = config.visionConfiguration
        let factor = config.pixelShuffleFactor

        self._visionModel.wrappedValue = SharpAILlamaNemotronVlVisionTower(vision)
        self._mlp1.wrappedValue = SharpAILlamaNemotronVlProjector.MLP1(
            inputDimensions: vision.hiddenSize * factor * factor,
            hiddenDimensions: text.hiddenSize,
            outputDimensions: text.hiddenSize)
        self._languageModel.wrappedValue = SharpAILlamaNemotronVlLanguage.Model(text)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAILlamaNemotronVlSanitize.sanitize(weights: weights)
    }

    /// Hidden states `[batch, seq, text hidden]`.
    ///
    /// - Parameters:
    ///   - inputIds: `[batch, seq]` token ids.
    ///   - pixelValues: `[images, channels, height, width]` NCHW pixels, if any. Each image needs
    ///     exactly `(image_size / patch_size / k)^2` rows of `imageTokenId` in `inputIds`, in order.
    public func embed(_ inputIds: MLXArray, pixelValues: MLXArray? = nil) throws -> MLXArray {
        let textEmbeds = languageModel.embedTokens(inputIds)
        let merged: MLXArray
        if let pixelValues {
            let features = imageFeatures(pixelValues, dtype: textEmbeds.dtype)
            merged = try mergeImageFeatures(features, into: textEmbeds, inputIds: inputIds)
        } else {
            merged = textEmbeds
        }
        return languageModel(inputEmbeddings: merged)
    }

    /// Mean over the sequence axis of `embed`. UNVERIFIED: not the reference pooling.
    public func meanPooled(
        _ inputIds: MLXArray, pixelValues: MLXArray? = nil
    ) throws -> MLXArray {
        try embed(inputIds, pixelValues: pixelValues).mean(axis: 1)
    }

    /// Text-only forward. Returns the hidden states in the logits slot.
    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        // The text-only path cannot throw; only image merging throws.
        try! embed(inputs)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        let hidden = try embed(tokens, pixelValues: input.image?.pixels)
        let total = tokens.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: hidden))
    }

    /// Vision features as `[images * tokens, text hidden]`, after pixel shuffle and projection.
    private func imageFeatures(_ pixelValues: MLXArray, dtype: DType) -> MLXArray {
        let nhwc = pixelValues.transposed(0, 2, 3, 1).asType(dtype)
        let patches = visionModel.transformer(nhwc)

        let count = patches.dim(1)
        let side = Int(Double(count).squareRoot())
        precondition(side * side == count, "vision tokens must form a square grid")

        let grid = patches.reshaped(patches.dim(0), side, side, patches.dim(2))
        let shuffled = SharpAILlamaNemotronVlProjector.pixelShuffle(
            grid, factor: config.pixelShuffleFactor, version: config.psVersion)
        let tokens = shuffled.reshaped(shuffled.dim(0), -1, shuffled.dim(3))
        let projected = mlp1(tokens)
        return projected.reshaped(-1, projected.dim(-1))
    }

    /// Writes image features into the image-token rows, in order.
    private func mergeImageFeatures(
        _ features: MLXArray, into textEmbeds: MLXArray, inputIds: MLXArray
    ) throws -> MLXArray {
        let hiddenSize = textEmbeds.dim(-1)
        let imageFlags = (inputIds .== MLXArray(config.imageTokenId)).asArray(Bool.self)
        let indices = imageFlags.enumerated().filter { $0.element }.map { Int32($0.offset) }

        let featureRows = features.dim(0)
        guard indices.count == featureRows else {
            throw SharpAILlamaNemotronVlError.imageTokenMismatch(
                expected: featureRows, actual: indices.count)
        }
        guard !indices.isEmpty else { return textEmbeds }

        let flat = textEmbeds.reshaped([-1, hiddenSize])
        flat[MLXArray(indices), 0...] = features.reshaped([featureRows, hiddenSize])
        return flat.reshaped(textEmbeds.shape)
    }
}
