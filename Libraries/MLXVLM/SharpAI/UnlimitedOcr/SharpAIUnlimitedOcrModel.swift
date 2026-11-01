// SharpAI fork overlay: unlimited_ocr model. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/baidu/Unlimited-OCR/raw/main/modeling_unlimitedocr.py
//   UnlimitedOCRModel.forward (global-view branch, the `else` path that runs with a 1024 px view),
//   and the masked_scatter that writes image features into the placeholder positions.
// Image placeholder id 128815 is `image_token_id` in UnlimitedOCRForCausalLM.infer.
//
// Global view layout (1024 px): the SAM + CLIP-L features of 16 x 16 tokens are projected to
// 2048 -> hidden, a newline token closes each row (16 x 17), and a view separator ends the image,
// giving 273 features. Crop tiles (the 640 px path) are not ported; see the vision file header.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The `model.*` namespace of the checkpoint: text decoder plus the vision stack and image tokens.
final class SharpAIUnlimitedOcrBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIUnlimitedOcrDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    @ModuleInfo(key: "sam_model") var samModel: SharpAIUnlimitedOcrSAMEncoder?
    @ModuleInfo(key: "vision_model") var visionModel: SharpAIUnlimitedOcrCLIPVision?
    @ModuleInfo(key: "projector") var projector: SharpAIUnlimitedOcrProjector?
    @ParameterInfo(key: "image_newline") var imageNewline: MLXArray
    @ParameterInfo(key: "view_seperator") var viewSeperator: MLXArray

    init(_ configuration: SharpAIUnlimitedOcrConfiguration) {
        let text = configuration.text
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: text.vocabSize, dimensions: text.hiddenSize)
        self.layers = (0 ..< text.numHiddenLayers).map {
            SharpAIUnlimitedOcrDecoderLayer(text, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: text.hiddenSize, eps: text.rmsNormEps)
        if configuration.hasVisionConfig {
            _samModel.wrappedValue = SharpAIUnlimitedOcrSAMEncoder()
            _visionModel.wrappedValue = SharpAIUnlimitedOcrCLIPVision()
            _projector.wrappedValue = SharpAIUnlimitedOcrProjector(outputDim: text.hiddenSize)
        }
        _imageNewline.wrappedValue = zeros([text.hiddenSize])
        _viewSeperator.wrappedValue = zeros([text.hiddenSize])
    }

    /// Runs the decoder over token embeddings. Full causal attention, RoPE offset from the cache.
    func forward(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }

    /// Token embeddings with the global-view image features written into the placeholder positions.
    /// `tokens` is (1, T); `pixels` is the 1024 px global view in NHWC layout, (1, 1024, 1024, 3).
    func embedImages(tokens: MLXArray, pixels: MLXArray) throws -> MLXArray {
        guard let samModel, let visionModel, let projector else {
            throw SharpAIUnlimitedOcrError.visionNotConfigured
        }
        let embeddings = embedTokens(tokens)
        let dim = embeddings.dim(2)

        // SAM feature map (1, g, g, 1024); CLIP-L consumes it as its patch embeddings.
        let sam = try samModel(pixels)
        let clip = try visionModel(sam)
        let grid = sam.dim(1)

        // [CLIP patch tokens (without CLS), SAM tokens] concatenated along channels, as in forward().
        let clipPatches = clip[0..., 1..., 0...]
        let samTokens = sam.reshaped(1, grid * grid, sam.dim(3))
        let projected = projector(concatenated([clipPatches, samTokens], axis: -1))

        // Each row ends with image_newline, then the whole image ends with view_seperator.
        let rows = projected.reshaped(grid, grid, dim)
        let newlines = repeated(imageNewline.reshaped(1, 1, dim), count: grid, axis: 0)
        let withNewlines = concatenated([rows, newlines], axis: 1).reshaped(-1, dim)
        let features = concatenated(
            [withNewlines, viewSeperator.reshaped(1, dim)], axis: 0
        ).asType(embeddings.dtype)

        // masked_scatter: the k-th placeholder position takes the k-th feature row.
        let imageMask = tokens .== MLXArray(Int32(SharpAIUnlimitedOcrModel.imageTokenId))
        let placeholders = imageMask.asType(.int32).sum(axis: -1).asArray(Int.self)
            .reduce(0, +)
        guard placeholders == features.dim(0) else {
            throw SharpAIUnlimitedOcrError.imageTokenCountMismatch(
                expected: features.dim(0), actual: placeholders)
        }
        let position = cumsum(imageMask.asType(.int32), axis: 1) - 1
        let clipped = maximum(position, MLXArray(Int32(0)))
        let gathered = take(features, clipped.reshaped(-1), axis: 0).reshaped(1, -1, dim)
        return which(imageMask.expandedDimensions(axis: -1), gathered, embeddings)
    }
}

/// unlimited_ocr: DeepSeek-V2 text decoder with a SAM + CLIP-L vision stack.
public class SharpAIUnlimitedOcrModel: Module, VLMModel, KVCacheDimensionProvider {
    /// `<image>` placeholder token id (`image_token_id` in UnlimitedOCRForCausalLM.infer).
    public static let imageTokenId = 128815

    public let configuration: SharpAIUnlimitedOcrConfiguration
    public var kvHeads: [Int]

    @ModuleInfo(key: "model") var model: SharpAIUnlimitedOcrBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: SharpAIUnlimitedOcrConfiguration) {
        self.configuration = configuration
        self.kvHeads = Array(
            repeating: configuration.text.numKeyValueHeads,
            count: configuration.text.numHiddenLayers)
        self._model.wrappedValue = SharpAIUnlimitedOcrBackbone(configuration)
        self._lmHead.wrappedValue = Linear(
            configuration.text.hiddenSize, configuration.text.vocabSize, bias: false)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model.forward(embeddings: model.embedTokens(inputs), cache: cache))
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard let image = input.image else {
            // Text-only prompts take the regular token path.
            return .tokens(input.text)
        }
        let tokens =
            input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        let embeddings = try model.embedImages(tokens: tokens, pixels: image.pixels)
        let hidden = model.forward(embeddings: embeddings, cache: cache.isEmpty ? nil : cache)
        let total = embeddings.dim(1)
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: lmHead(hidden)))
    }

    /// Stacks per-expert checkpoint tensors into the `switch_mlp` layout the MoE layers use.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = SharpAIUnlimitedOcrSanitizer.sanitize(weights)
        let text = configuration.text
        for layer in 0 ..< text.numHiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let experts = (0 ..< text.nRoutedExperts).map {
                        "\(prefix).experts.\($0).\(projection).\(suffix)"
                    }
                    guard !experts.isEmpty, experts.allSatisfy({ result[$0] != nil }) else {
                        continue
                    }
                    let joined = stacked(experts.map { result.removeValue(forKey: $0)! })
                    result["\(prefix).switch_mlp.\(projection).\(suffix)"] = joined
                }
            }
        }
        return result
    }
}
