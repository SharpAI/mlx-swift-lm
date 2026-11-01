// SharpAI fork overlay: modernbert encoder. Not part of upstream mlx-swift-lm.
//
// Module tree and weight names follow HF ModernBertModel:
//   model.embeddings.tok_embeddings, model.embeddings.norm,
//   model.layers.N.attn.Wqkv, model.layers.N.attn.Wo, model.layers.N.attn_norm (absent in layer 0),
//   model.layers.N.mlp_norm, model.layers.N.mlp.Wi, model.layers.N.mlp.Wo, model.final_norm.
//
// Layers are pre-norm and bidirectional. Layer N is global when N % global_attn_every_n_layers
// is 0. Global layers attend to the whole input. Other layers attend within a local window.
// No KV cache: each call runs the full input once.
//
// UNVERIFIED: these details come from memory of the HF modeling code, not from a source in this
// repo: layer 0 has no attn_norm; the MLP chunks Wi output as (activation input, gate) with the
// activation on the first half; the local window is |i - j| <= local_attention / 2; RoPE uses the
// rotate-half form with theta chosen per layer type.
// UNVERIFIED: the ColBERT projection (pylate `linear`, 768 -> colbert dim) is not implemented.
// Its checkpoint key and dimension are not in this repo. Outputs are the 768-d hidden states.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum SharpAIModernbertMasks {

    /// Bool mask [1, 1, L, L]. True where |i - j| <= halfWidth.
    static func slidingWindow(length: Int, halfWidth: Int) -> MLXArray {
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let distance = (positions.reshaped(length, 1) - positions.reshaped(1, length)).abs()
        return (distance .<= Int32(halfWidth)).reshaped(1, 1, length, length)
    }
}

final class SharpAIModernbertAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: SharpAIModernbertConfiguration, ropeTheta: Float) {
        heads = config.attentionHeads
        headDim = config.headDimensions
        scale = pow(Float(headDim), -0.5)

        self._wqkv.wrappedValue = Linear(
            config.hiddenSize, 3 * config.hiddenSize, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(
            config.hiddenSize, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: ropeTheta, traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = wqkv(x).reshaped(B, L, 3, heads, headDim)
        let parts = qkv.split(parts: 3, axis: 2)

        var queries = parts[0].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: nil)
        keys = applyRotaryPosition(rope, to: keys, offset: nil)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class SharpAIModernbertMLP: Module {

    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ config: SharpAIModernbertConfiguration) {
        self._wi.wrappedValue = Linear(
            config.hiddenSize, 2 * config.intermediateSize, bias: config.mlpBias)
        self._wo.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    /// GeGLU: the first half goes through GELU and multiplies the second half.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let halves = wi(x).split(parts: 2, axis: -1)
        return wo(gelu(halves[0]) * halves[1])
    }
}

final class SharpAIModernbertLayer: Module {

    let isGlobal: Bool

    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm?
    @ModuleInfo(key: "attn") var attn: SharpAIModernbertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIModernbertMLP

    init(_ config: SharpAIModernbertConfiguration, layerIndex: Int) {
        let global = layerIndex % config.globalAttnEveryNLayers == 0
        isGlobal = global

        self._attnNorm.wrappedValue =
            layerIndex == 0
            ? nil
            : LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps, affine: true,
                bias: config.normBias)
        self._attn.wrappedValue = SharpAIModernbertAttention(
            config, ropeTheta: global ? config.globalRopeTheta : config.localRopeTheta)
        self._mlpNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true,
            bias: config.normBias)
        self._mlp.wrappedValue = SharpAIModernbertMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let normed = attnNorm.map { $0(x) } ?? x
        let h = x + attn(normed, mask: mask)
        return h + mlp(mlpNorm(h))
    }
}

final class SharpAIModernbertEmbeddings: Module {

    @ModuleInfo(key: "tok_embeddings") var tokEmbeddings: Embedding
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: SharpAIModernbertConfiguration) {
        self._tokEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true,
            bias: config.normBias)
    }

    func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        norm(tokEmbeddings(inputs))
    }
}

final class SharpAIModernbertInner: Module {

    @ModuleInfo(key: "embeddings") var embeddings: SharpAIModernbertEmbeddings

    let layers: [SharpAIModernbertLayer]
    let localHalfWindow: Int

    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm

    init(_ config: SharpAIModernbertConfiguration) {
        self._embeddings.wrappedValue = SharpAIModernbertEmbeddings(config)
        self.layers = (0 ..< config.hiddenLayers).map { index in
            SharpAIModernbertLayer(config, layerIndex: index)
        }
        self.localHalfWindow = config.localHalfWindow
        self._finalNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEps, affine: true,
            bias: config.normBias)
    }

    /// `padding` is an optional Bool mask [B, 1, 1, L]. True marks a real token.
    func callAsFunction(_ inputs: MLXArray, padding: MLXArray?) -> MLXArray {
        var h = embeddings(inputs)
        let length = h.dim(1)

        let window = SharpAIModernbertMasks.slidingWindow(
            length: length, halfWidth: localHalfWindow)

        let globalMode: MLXFast.ScaledDotProductAttentionMaskMode
        let localMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let padding {
            globalMode = .array(padding)
            localMode = .array(window & padding)
        } else {
            globalMode = .none
            localMode = .array(window)
        }

        for layer in layers {
            h = layer(h, mask: layer.isGlobal ? globalMode : localMode)
        }
        return finalNorm(h)
    }
}

/// Bidirectional ModernBERT encoder for `modernbert`.
final class SharpAIModernbertModel: Module, LLMModel, LoRAModel {

    @ModuleInfo(key: "model") var model: SharpAIModernbertInner

    let configuration: SharpAIModernbertConfiguration

    init(_ configuration: SharpAIModernbertConfiguration) {
        self.configuration = configuration
        self._model.wrappedValue = SharpAIModernbertInner(configuration)
    }

    /// Final hidden states, shape [B, L, hidden]. `attentionMask` is [B, L] with 1 for real
    /// tokens and 0 for padding. Padding keys are excluded from every layer.
    func hiddenStates(_ inputs: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let padding = attentionMask.map { mask in
            mask.asType(.bool).reshaped(inputs.dim(0), 1, 1, inputs.dim(1))
        }
        return model(inputs, padding: padding)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        hiddenStates(inputs)
    }

    /// The encoder has no cache, so each call sees the whole input.
    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        []
    }

    /// Chunked prefill would hide later tokens from earlier positions, so prepare must refuse.
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        throw SharpAIModernbertError.generationUnsupported
    }

    func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        SharpAIModernbertSanitizer.sanitize(weights)
    }

    var loraLayers: [Module] {
        model.layers
    }
}
