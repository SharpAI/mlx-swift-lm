// SharpAI fork overlay: moondream3 vision-language model. Not part of upstream mlx-swift-lm.
//
// The image span is the first `1 + enc_n_layers^2` prompt positions: a BOS row, then the
// projected vision rows. `prepare` replaces that span with those rows, as vLLM's
// embed_multimodal does. Logits for the answer delimiter are suppressed, as in vLLM.
// UNVERIFIED: no real checkpoint was run. Forward shapes are checked only by tiny tests.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public class SharpAIMoondream3Model: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision") private var vision: SharpAIMoondream3Vision
    @ModuleInfo(key: "text") private var textModel: SharpAIMoondream3TextModel

    public let configuration: SharpAIMoondream3Configuration

    public var vocabularySize: Int { configuration.textConfiguration.vocabularySize }

    public var kvHeads: [Int] {
        Array(
            repeating: configuration.textConfiguration.kvHeads,
            count: configuration.textConfiguration.layers)
    }

    public var loraLayers: [Module] {
        textModel.blocks
    }

    public init(_ configuration: SharpAIMoondream3Configuration) {
        self.configuration = configuration
        _vision.wrappedValue = SharpAIMoondream3Vision(
            configuration.visionConfiguration, textDim: configuration.textConfiguration.dim)
        _textModel.wrappedValue = SharpAIMoondream3TextModel(configuration.textConfiguration)
        super.init()
    }

    /// BOS row followed by the projected vision rows, `[1 + P, text_dim]`.
    private func imagePrefix(_ image: LMInput.ProcessedImage) throws -> MLXArray {
        let pixels = image.pixels
        let crop = configuration.visionConfiguration.cropSize
        guard pixels.ndim == 4, pixels.dim(1) == 3, pixels.dim(2) == crop, pixels.dim(3) == crop
        else {
            throw SharpAIMoondream3Error.imageShape(pixels.shape)
        }

        var tiling: (rows: Int, cols: Int)? = nil
        if let frame = image.frames?.first {
            tiling = (frame.h, frame.w)
        }

        let rows = try vision.imageRows(pixels, tiling: tiling)
        let bos = textModel.wte(MLXArray([Int32(SharpAIMoondream3Configuration.bosTokenId)]))
        return MLX.concatenated([bos.asType(rows.dtype), rows], axis: 0)
    }

    private func suppressAnswer(_ logits: MLXArray) -> MLXArray {
        let answer = configuration.answerTokenId
        guard answer >= 0, answer < vocabularySize else {
            return logits
        }
        var out = logits
        out[.ellipsis, answer] = MLXArray(-Float.infinity).asType(logits.dtype)
        return out
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
        var embeddings = textModel.wte(tokens)

        if let image = input.image {
            guard tokens.dim(0) == 1 else {
                throw SharpAIMoondream3Error.batchedInputUnsupported(tokens.dim(0))
            }
            let prefixLength = configuration.imagePrefixLength
            guard tokens.dim(-1) >= prefixLength else {
                throw SharpAIMoondream3Error.promptTooShort(
                    tokens: tokens.dim(-1), prefix: prefixLength)
            }
            let rows = try imagePrefix(image)
            let rest = embeddings[0..., prefixLength..., 0...]
            embeddings = MLX.concatenated(
                [rows.expandedDimensions(axis: 0).asType(embeddings.dtype), rest], axis: 1)
        }

        let logits = suppressAnswer(textModel(embeddings: embeddings, cache: cache))
        let total = tokens.dim(-1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        suppressAnswer(textModel(embeddings: textModel.wte(inputs), cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIMoondream3Sanitizer.sanitize(weights)
    }
}
