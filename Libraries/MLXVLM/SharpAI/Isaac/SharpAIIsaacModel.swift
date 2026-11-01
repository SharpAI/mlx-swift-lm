// SharpAI fork overlay: isaac model (vision tower, projector, Qwen3-style text decoder).
// Not part of upstream mlx-swift-lm.
//
// Ported from Perceptron's modular_isaac.py (IsaacModel, IsaacMultiModalProjector) and the
// Qwen3 decoder it reuses. Checkpoint names are produced by SharpAIIsaacSanitizer.
//
// Scope, stated plainly:
// - Text-only forward is implemented. With no image, positions are sequential and the
//   3-axis mRoPE of Isaac reduces to plain RoPE, so this matches the reference.
// - Image features: `imageFeatures(_:)` runs tower -> pixel shuffle -> projector for a fixed
//   square grid. Splicing image features into the token stream is NOT implemented.
// - NaFlex (variable resolution, packed sequences, document masks) is NOT implemented.
// - Weights are not loaded in tests; tests check shapes and checkpoint key coverage.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Projector

/// `model.multimodal_projector`: pixel-shuffled vision features to the text hidden size.
/// Both linears are bias-free; the activation is SiLU.
final class SharpAIIsaacMultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionHidden: Int, textHidden: Int) {
        _linear1.wrappedValue = Linear(visionHidden, 4 * visionHidden, bias: false)
        _linear2.wrappedValue = Linear(4 * visionHidden, textHidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

// MARK: - Text decoder (Qwen3 layout: q_norm and k_norm, RoPE theta from config)

final class SharpAIIsaacTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: SharpAIIsaacTextConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: nil)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(batch, length, heads, -1))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)

        return oProj(output)
    }
}

final class SharpAIIsaacTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAIIsaacTextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIIsaacTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIIsaacTextAttention
    let mlp: SharpAIIsaacTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIIsaacTextConfiguration) {
        _attention.wrappedValue = SharpAIIsaacTextAttention(config)
        self.mlp = SharpAIIsaacTextMLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// `model.text_model`: embeddings, decoder layers and the final norm.
final class SharpAIIsaacTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIIsaacTextDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIIsaacTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIIsaacTextDecoderLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

// MARK: - Composite model

/// Isaac conditional-generation model. Parameter paths match the sanitized checkpoint names.
public final class SharpAIIsaacModel: Module, LanguageModel, KVCacheDimensionProvider {
    public let kvHeads: [Int]
    let visionConfiguration: SharpAIIsaacVisionConfiguration

    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIIsaacVisionModel
    @ModuleInfo(key: "multimodal_projector") var multimodalProjector:
        SharpAIIsaacMultiModalProjector
    @ModuleInfo(key: "text_model") var textModel: SharpAIIsaacTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: SharpAIIsaacConfiguration) {
        let text = configuration.textConfiguration
        let vision = configuration.visionConfiguration
        self.kvHeads = Array(repeating: text.kvHeads, count: text.hiddenLayers)
        self.visionConfiguration = vision

        _visionTower.wrappedValue = SharpAIIsaacVisionModel(vision)
        _multimodalProjector.wrappedValue = SharpAIIsaacMultiModalProjector(
            visionHidden: vision.hiddenSize * vision.pixelShuffleScaleFactor
                * vision.pixelShuffleScaleFactor,
            textHidden: text.hiddenSize)
        _textModel.wrappedValue = SharpAIIsaacTextModel(text)
        _lmHead.wrappedValue = Linear(text.hiddenSize, text.vocabularySize, bias: false)
    }

    /// Prefill the prompt into `cache` and return logits for the final position.
    ///
    /// Text-only: image or video input throws `SharpAIIsaacError.visionNotImplemented`
    /// because splicing image features into the token stream is not implemented.
    /// All positions but the last run through the text decoder without `lm_head`, so the
    /// vocabulary projection is computed only once, for the position that is sampled.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw SharpAIIsaacError.visionNotImplemented
        }
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let totalTokens = tokens.dim(1)
        guard totalTokens > 0 else {
            throw SharpAIIsaacError.emptyPrompt
        }

        let processed = try prefill.forEachChunk(total: totalTokens) { range in
            _ = textModel(tokens[0..., range], cache: cache)
            asyncEval(cache)
        }

        let logits = lmHead(textModel(tokens[0..., processed...], cache: cache))
        prefill.progress?(totalTokens, totalTokens)
        return .logits(LMOutput(logits: logits))
    }

    /// Text logits for token ids `[B, L]`.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(textModel(inputs, cache: cache))
    }

    /// Image patches `[N, P, C * p * p]` (fixed grid, P == grid²) to text-sized features
    /// `[N, (grid / s)², text_hidden]`.
    public func imageFeatures(_ patches: MLXArray) -> MLXArray {
        let hidden = visionTower(patches)
        let shuffled = SharpAIIsaacPixelShuffle.shuffle(
            hidden,
            gridSide: visionConfiguration.gridSide,
            scale: visionConfiguration.pixelShuffleScaleFactor)
        return multimodalProjector(shuffled)
    }

    /// Maps checkpoint names to parameter paths and rejects duplicates.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try SharpAIIsaacSanitizer.sanitize(weights)
    }
}
