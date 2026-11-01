// SharpAI fork overlay: OpenVLA vision-language model. Not part of upstream mlx-swift-lm.
//
// Fused DINOv2 + SigLIP features go through a 3-layer GELU MLP into the Llama-2 embedding
// width. The projected patches are placed after the BOS token. Action-token decoding is not
// implemented here. There is no image processor, so callers pass normalized NCHW pixels with
// 6 channels: DINOv2 input in channels 0..<3 and SigLIP input in channels 3..<6.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Two-layer-plus-output MLP from fused vision width to text width.
/// Module path `projector.fc1`, `fc2`, `fc3`.
/// UNVERIFIED: the first hidden width is 4x the fused vision width. This is recalled from
/// the reference code. The checkpoint shapes are the only check.
final class SharpAIOpenvlaProjector: Module {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo var fc3: Linear

    init(visionDim: Int, textDim: Int) {
        let hidden = 4 * visionDim
        self._fc1.wrappedValue = Linear(visionDim, hidden, bias: true)
        self._fc2.wrappedValue = Linear(hidden, textDim, bias: true)
        self._fc3.wrappedValue = Linear(textDim, textDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc3(gelu(fc2(gelu(fc1(x)))))
    }
}

/// OpenVLA. Weight keys follow HF names after `SharpAIOpenvlaSanitize`:
/// `vision_backbone.*`, `projector.*`, `language_model.model.*`, `language_model.lm_head`.
public final class SharpAIOpenvla: Module, VLMModel, KVCacheDimensionProvider {

    public let config: SharpAIOpenvlaConfiguration
    let visionSpec: SharpAIOpenvlaVisionSpec

    @ModuleInfo(key: "vision_backbone") var visionBackbone: SharpAIOpenvlaVisionBackbone
    @ModuleInfo var projector: SharpAIOpenvlaProjector
    @ModuleInfo(key: "language_model") var languageModel: SharpAIOpenvlaLanguageModel

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public convenience init(_ config: SharpAIOpenvlaConfiguration) {
        self.init(config, vision: .openvla7b)
    }

    /// The vision spec is a parameter so tests can build a tiny model. Production uses `.openvla7b`.
    init(_ config: SharpAIOpenvlaConfiguration, vision: SharpAIOpenvlaVisionSpec) {
        self.config = config
        self.visionSpec = vision
        self._visionBackbone.wrappedValue = SharpAIOpenvlaVisionBackbone(vision)
        self._projector.wrappedValue = SharpAIOpenvlaProjector(
            visionDim: vision.fusedHiddenSize, textDim: config.textConfiguration.hiddenSize)
        self._languageModel.wrappedValue = SharpAIOpenvlaLanguageModel(
            config.textConfiguration)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIOpenvlaSanitize.sanitize(weights: weights)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let tokens = input.text.tokens
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

    /// NCHW pixels `[batch, 6, H, W]` to projected patch tokens `[batch, patches, text hidden]`.
    func imageFeatures(_ pixelValues: MLXArray, dtype: DType) throws -> MLXArray {
        guard pixelValues.ndim == 4 else {
            throw SharpAIOpenvlaError.imageChannelMismatch(pixelValues.ndim)
        }
        guard pixelValues.dim(1) == 6 else {
            throw SharpAIOpenvlaError.imageChannelMismatch(pixelValues.dim(1))
        }
        let nhwc = pixelValues.transposed(0, 2, 3, 1).asType(dtype)
        let visual = try visionBackbone(nhwc)
        return projector(visual)
    }

    /// `[BOS, image patches, remaining text]` embeddings. The tokens must start with BOS.
    /// UNVERIFIED: the image slot after BOS is recalled from the reference code.
    func multimodalEmbeddings(_ tokens: MLXArray, pixelValues: MLXArray) throws -> MLXArray {
        let textEmbeds = languageModel.embed(tokens)
        let features = try imageFeatures(pixelValues, dtype: textEmbeds.dtype)
        guard features.dim(0) == textEmbeds.dim(0) else {
            throw SharpAIOpenvlaError.batchMismatch(
                text: textEmbeds.dim(0), image: features.dim(0))
        }
        let bos = textEmbeds[0..., 0 ..< 1, 0...]
        let rest = textEmbeds[0..., 1..., 0...]
        return concatenated([bos, features, rest], axis: 1)
    }
}
