// SharpAI fork overlay: phi4-siglip vision-language model. Not part of upstream mlx-swift-lm.
//
// NaFlex vision tower, then a two-layer GELU projector (`mlp2x_gelu`), then the Phi-style text
// decoder. Each image patch row becomes one token, and the features replace the rows where the
// prompt holds `image_token_index`. Created by ``SharpAIPhi4SiglipOverlay``.
// UNVERIFIED: the one-token-per-patch mapping and the projector key names (`0`, `2`).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// `mlp2x_gelu` projector: Linear, GELU, Linear. Keys are `mm_projector.0` and `mm_projector.2`.
final class SharpAIPhi4SiglipProjector: Module {
    @ModuleInfo(key: "0") var linear0: Linear
    @ModuleInfo(key: "2") var linear2: Linear

    init(inputSize: Int, hiddenSize: Int) {
        _linear0.wrappedValue = Linear(inputSize, hiddenSize, bias: true)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(MLXNN.gelu(linear0(x)))
    }
}

/// phi4-siglip model. Created by ``SharpAIPhi4SiglipOverlay``.
public class SharpAIPhi4SiglipModel: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo var model: SharpAIPhi4SiglipBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let configuration: SharpAIPhi4SiglipConfiguration

    public var vocabularySize: Int { configuration.vocabularySize }

    public var kvHeads: [Int] {
        Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public init(_ configuration: SharpAIPhi4SiglipConfiguration) {
        self.configuration = configuration
        _model.wrappedValue = SharpAIPhi4SiglipBackbone(configuration)
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
    }

    /// Image features for one call, one tower pass per image in `frames` order.
    /// Returns `[total patch rows, llm hidden]`.
    private func imageFeatures(_ image: LMInput.ProcessedImage) throws -> MLXArray {
        guard let frames = image.frames, !frames.isEmpty else {
            throw SharpAIPhi4SiglipError.missingImageGrid
        }

        var offset = 0
        var blocks: [MLXArray] = []
        for frame in frames {
            guard frame.t == 1, frame.h > 0, frame.w > 0 else {
                throw SharpAIPhi4SiglipError.invalidImageFrame
            }
            let count = frame.h * frame.w
            guard offset + count <= image.pixels.dim(0) else {
                throw SharpAIPhi4SiglipError.imageGridMismatch(
                    expected: offset + count, actual: image.pixels.dim(0))
            }
            let patches = image.pixels[offset ..< (offset + count)]
            offset += count

            let hidden = model.visionTower(patches, gridHeight: frame.h, gridWidth: frame.w)
            blocks.append(model.mmProjector(hidden))
        }

        guard offset == image.pixels.dim(0) else {
            throw SharpAIPhi4SiglipError.imageGridMismatch(
                expected: offset, actual: image.pixels.dim(0))
        }
        return concatenated(blocks, axis: 0)
    }

    /// Writes image features into the rows where `tokens == image_token_index`, in order.
    private func scatterImageFeatures(
        _ features: MLXArray, into embeddings: MLXArray, tokens: MLXArray
    ) throws -> MLXArray {
        guard tokens.ndim == 2, tokens.dim(0) == 1 else {
            throw SharpAIPhi4SiglipError.batchedImageUnsupported(tokens.dim(0))
        }
        guard let imageTokenIndex = configuration.resolvedImageTokenIndex else {
            throw SharpAIPhi4SiglipError.imageTokenIndexMissing
        }

        let ids = tokens.asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == imageTokenIndex }
        guard positions.count == features.dim(0) else {
            throw SharpAIPhi4SiglipError.imagePlaceholderMismatch(
                expected: positions.count, actual: features.dim(0))
        }
        guard !positions.isEmpty else { return embeddings }

        let hidden = embeddings.dim(-1)
        let flat = embeddings.reshaped([-1, hidden])
        flat[MLXArray(positions.map { Int32($0) }), 0...] =
            features.asType(flat.dtype).reshaped([positions.count, hidden])
        return flat.reshaped(embeddings.shape)
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        var embeddings = model.embedTokens(tokens)

        if let image = input.image {
            let features = try imageFeatures(image)
            embeddings = try scatterImageFeatures(features, into: embeddings, tokens: tokens)
        }

        let hidden = model(embeddings: embeddings, cache: cache)
        let total = tokens.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits(hidden)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model(embeddings: model.embedTokens(inputs), cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIPhi4SiglipSanitizer.sanitize(weights, configuration: configuration)
    }
}
