// SharpAI fork overlay: qianfan_ocr (InternViT + Qwen3) vision-language model. Not part of upstream.
// Image path: vision tower, drop CLS, pixel shuffle (2x2 patch tokens to one token), MLP projector,
// then the image features replace the image placeholder ids in the text embeddings.
// No processor is registered: resize, normalization and the `<image>` expansion are not written.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Space-to-depth over a square token grid, as the reference `pixel_shuffle`.
/// `[batch, H, W, C]` becomes `[batch, H / r, W / r, C * r * r]`, and channels inside a block
/// are ordered (row, column, channel). The shape steps mirror the reference view and permute.
func sharpAIQianfanOcrPixelShuffle(_ x: MLXArray, factor r: Int) -> MLXArray {
    let (batch, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    var y = x.reshaped(batch, h, w / r, c * r)
    y = y.transposed(0, 2, 1, 3)
    y = y.reshaped(batch, w / r, h / r, c * r * r)
    return y.transposed(0, 2, 1, 3)
}

/// `LayerNorm` then `Linear`, `GELU`, `Linear`. Checkpoint names are `mlp1.0`, `mlp1.1`, `mlp1.3`.
final class SharpAIQianfanOcrProjector: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimensions: Int, textHidden: Int) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: inputDimensions)
        _linear1.wrappedValue = Linear(inputDimensions, textHidden)
        _linear2.wrappedValue = Linear(textHidden, textHidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.gelu(linear1(layerNorm(x))))
    }
}

public class SharpAIQianfanOcr: Module, VLMModel, KVCacheDimensionProvider {
    public let config: SharpAIQianfanOcrConfiguration

    @ModuleInfo(key: "vision_tower") private var visionTower: SharpAIQianfanOcrVisionTower
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        SharpAIQianfanOcrProjector
    @ModuleInfo(key: "language_model") private var languageModel: SharpAIQianfanOcrLanguageModel

    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: SharpAIQianfanOcrConfiguration) {
        self.config = config
        let vision = config.visionConfiguration
        let factor = config.downsampleFactor
        self._visionTower.wrappedValue = SharpAIQianfanOcrVisionTower(vision)
        self._multiModalProjector.wrappedValue = SharpAIQianfanOcrProjector(
            inputDimensions: vision.hiddenSize * factor * factor,
            textHidden: config.textConfiguration.hiddenSize)
        self._languageModel.wrappedValue = SharpAIQianfanOcrLanguageModel(config.textConfiguration)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }
        guard tokens.dim(0) == 1 else {
            throw SharpAIQianfanOcrError.invalidInput("batch size must be 1, got \(tokens.dim(0))")
        }

        let embeddings = try inputEmbeddings(tokens: tokens, image: input.image)
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel.logits(embeddings: embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        let result = languageModel.logits(
            embeddings: embeddings[0..., processed..., 0...], cache: cache)
        prefill.progress?(totalPositions, totalPositions)
        return .logits(LMOutput(logits: result))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel.logits(embeddings: languageModel.embed(inputs), cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIQianfanOcrSanitize.sanitize(weights: weights, config: config)
    }

    // MARK: - Image

    private func inputEmbeddings(tokens: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let textEmbeddings = languageModel.embed(tokens)
        guard let image else { return textEmbeddings }

        let imageEmbeddings = try encodeImages(image.pixels)
        return try mergeImages(
            textEmbeddings: textEmbeddings,
            tokenIds: tokens[0].asArray(Int.self),
            imageEmbeddings: imageEmbeddings)
    }

    /// Encodes `[images, H, W, C]` pixels. Returns `[images * tokensPerImage, textHidden]`.
    /// UNVERIFIED: no processor exists for this model, so the channel-last pixel layout and the
    /// ImageNet normalization that the reference applies are assumed, not checked.
    private func encodeImages(_ pixels: MLXArray) throws -> MLXArray {
        let vision = config.visionConfiguration
        guard pixels.ndim == 4, pixels.dim(1) == vision.imageSize,
            pixels.dim(2) == vision.imageSize, pixels.dim(3) == vision.numChannels
        else {
            throw SharpAIQianfanOcrError.invalidInput(
                "pixels must be [images, \(vision.imageSize), \(vision.imageSize), \(vision.numChannels)]"
            )
        }

        let images = pixels.dim(0)
        let patchFeatures = visionTower(pixels)[0..., 1..., 0...]
        let grid = vision.imageSize / vision.patchSize
        let grid2d = patchFeatures.reshaped(images, grid, grid, vision.hiddenSize)
        let shuffled = sharpAIQianfanOcrPixelShuffle(grid2d, factor: config.downsampleFactor)
        let tokens = shuffled.reshaped(images, -1, shuffled.dim(-1))
        let projected = multiModalProjector(tokens)
        return projected.reshaped(-1, projected.dim(-1))
    }

    /// Replaces each run of `imageTokenId` placeholders with the next image embeddings, in order.
    private func mergeImages(
        textEmbeddings: MLXArray, tokenIds: [Int], imageEmbeddings: MLXArray
    ) throws -> MLXArray {
        let imageTokenId = config.imageTokenId
        let placeholders = tokenIds.filter { $0 == imageTokenId }.count
        guard placeholders == imageEmbeddings.dim(0) else {
            throw SharpAIQianfanOcrError.imageTokenMismatch(
                placeholders: placeholders, features: imageEmbeddings.dim(0))
        }

        let images = imageEmbeddings.asType(textEmbeddings.dtype)
        var parts: [MLXArray] = []
        var textStart = 0
        var imageOffset = 0
        var index = 0
        while index < tokenIds.count {
            guard tokenIds[index] == imageTokenId else {
                index += 1
                continue
            }
            if textStart < index {
                parts.append(textEmbeddings[0..., textStart ..< index, 0...])
            }
            var runEnd = index
            while runEnd < tokenIds.count, tokenIds[runEnd] == imageTokenId {
                runEnd += 1
            }
            let runLength = runEnd - index
            parts.append(
                images[imageOffset ..< (imageOffset + runLength)].expandedDimensions(axis: 0))
            imageOffset += runLength
            textStart = runEnd
            index = runEnd
        }
        if textStart < tokenIds.count {
            parts.append(textEmbeddings[0..., textStart ..< tokenIds.count, 0...])
        }
        return concatenated(parts, axis: 1)
    }
}
