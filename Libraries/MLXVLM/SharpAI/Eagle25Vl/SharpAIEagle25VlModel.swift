// SharpAI fork overlay: eagle_2_5_vl (Eagle2.5-VL) model. Not part of upstream mlx-swift-lm.
//
// SigLIP tower, then pixel shuffle (factor 1 / downsample_ratio), then a LayerNorm + 2-layer GELU
// MLP into the Qwen2 embedding space. Image features replace the `<IMG_CONTEXT>` token rows.
// Mirrors vLLM's eagle2_5_vl.py: extract_feature, pixel_shuffle, and mlp1.
// UNVERIFIED: batch size 1 only, and no padding mask in prepare.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Projector `mlp1`: LayerNorm, Linear, GELU, Linear.
final class SharpAIEagle25VlProjector: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputSize: Int, hiddenSize: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: inputSize)
        _linear1.wrappedValue = Linear(inputSize, hiddenSize, bias: true)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.gelu(linear1(norm(x))))
    }
}

/// Eagle2.5-VL vision-language model. Created by ``SharpAIEagle25VlOverlay``.
public class SharpAIEagle25VlModel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_model") private var visionModel: SharpAIEagle25VlVisionModel
    @ModuleInfo(key: "mlp1") private var projector: SharpAIEagle25VlProjector
    @ModuleInfo(key: "language_model") private var languageModel:
        SharpAIEagle25VlTextLanguageModel

    public let configuration: SharpAIEagle25VlConfiguration

    public var vocabularySize: Int { configuration.textConfiguration.vocabularySize }

    public var kvHeads: [Int] {
        Array(
            repeating: configuration.textConfiguration.kvHeads,
            count: configuration.textConfiguration.hiddenLayers)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ configuration: SharpAIEagle25VlConfiguration) {
        self.configuration = configuration
        _visionModel.wrappedValue = SharpAIEagle25VlVisionModel(
            configuration.visionConfiguration, layerCount: configuration.visionLayerCount,
            applyPostLayerNorm: configuration.appliesVisionPostLayerNorm)
        _projector.wrappedValue = SharpAIEagle25VlProjector(
            inputSize: configuration.projectorInputSize,
            hiddenSize: configuration.textConfiguration.hiddenSize)
        _languageModel.wrappedValue = SharpAIEagle25VlTextLanguageModel(
            configuration.textConfiguration)
    }

    /// Rearranges `[N, W, H, C]` so each group of `scale`-by-`scale` patches becomes one token.
    /// Port of vLLM's `pixel_shuffle`.
    static func pixelShuffle(_ x: MLXArray, scale: Float) -> MLXArray {
        let (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let scaledHeight = Int(Float(h) * scale)
        let scaledChannels = Int(Float(c) / scale)
        let scaledWidth = Int(Float(w) * scale)
        let finalChannels = Int(Float(c) / (scale * scale))

        let first = x.reshaped(n, w, scaledHeight, scaledChannels).transposed(0, 2, 1, 3)
        return first.reshaped(n, scaledHeight, scaledWidth, finalChannels).transposed(0, 2, 1, 3)
    }

    /// Image features for `pixels` `[T, C, H, W]` (PyTorch layout). Returns `[T * seq, llmHidden]`.
    private func imageFeatures(_ pixels: MLXArray) -> MLXArray {
        let dtype = visionModel.visionModel.embeddings.patchEmbedding.weight.dtype
        let channelsLast = pixels.transposed(0, 2, 3, 1).asType(dtype)

        let hidden = visionModel(channelsLast)
        let side = Int(Double(hidden.dim(1)).squareRoot())
        let grid = hidden.reshaped(hidden.dim(0), side, side, hidden.dim(-1))

        let shuffled = Self.pixelShuffle(grid, scale: configuration.downsampleRatio)
        let tokens = shuffled.reshaped(shuffled.dim(0), -1, shuffled.dim(-1))
        return projector(tokens).reshaped(-1, configuration.textConfiguration.hiddenSize)
    }

    /// Writes image features into the rows where `tokens == imageTokenId`, in order.
    private func scatterImageFeatures(
        _ features: MLXArray, into embeddings: MLXArray, tokens: MLXArray, imageTokenId: Int
    ) throws -> MLXArray {
        guard tokens.ndim == 2, tokens.dim(0) == 1 else {
            throw SharpAIEagle25VlError.batchedImageUnsupported(tokens.dim(0))
        }

        let ids = tokens.asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == imageTokenId }
        guard positions.count == features.dim(0) else {
            throw SharpAIEagle25VlError.imageTokenCountMismatch(
                expected: positions.count, actual: features.dim(0))
        }
        guard !positions.isEmpty else { return embeddings }

        let hidden = embeddings.dim(-1)
        let flat = embeddings.reshaped([-1, hidden])
        flat[MLXArray(positions.map { Int32($0) }), 0...] =
            features.asType(flat.dtype).reshaped([positions.count, hidden])
        return flat.reshaped(embeddings.shape)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        var embeddings = languageModel.model.embedTokens(tokens)

        if let image = input.image {
            guard let imageTokenId = configuration.imageTokenId else {
                throw SharpAIEagle25VlError.imageTokenIdMissing
            }
            let features = imageFeatures(image.pixels)
            embeddings = try scatterImageFeatures(
                features, into: embeddings, tokens: tokens, imageTokenId: imageTokenId)
        }

        let logits = languageModel(embeddings, cache: cache)
        let total = tokens.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(languageModel.model.embedTokens(inputs), cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIEagle25VlSanitizer.sanitize(weights, configuration: configuration)
    }
}
