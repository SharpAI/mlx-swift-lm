// SharpAI fork overlay: voxtral_realtime text decoder with the delay-conditioned ada norm.
// Not part of upstream mlx-swift-lm.
//
// Each layer is pre-norm GQA attention (KV cached, RoPE, sliding window), then an MLP whose
// input is scaled by a delay-conditioned term: x * (1 + W_out gelu(W_in t)).
// UNVERIFIED: the ada norm placement (after the post-attention norm), its bias-free projections,
// its gelu, and the sinusoidal delay embedding are taken from the transformers layout as
// recalled, not checked against a checkpoint.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Causal self-attention with grouped KV heads and a KV cache.
final class SharpAIVoxtralRealtimeAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIVoxtralRealtimeTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = initializeRope(
            dims: config.headDim, base: config.ropeTheta, traditional: false,
            scalingConfig: nil, maxPositionEmbeddings: config.maxPositionEmbeddings)

        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

/// Scales the FFN input by a term that depends on the delay condition, not on the token.
final class SharpAIVoxtralRealtimeAdaNorm: Module {

    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(hidden: Int, dimension: Int) {
        self._linearIn.wrappedValue = Linear(hidden, dimension, bias: false)
        self._linearOut.wrappedValue = Linear(dimension, hidden, bias: false)
        super.init()
    }

    /// `x` is (batch, tokens, hidden). `condition` is (1, 1, hidden).
    func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
        x * (1 + linearOut(gelu(linearIn(condition))))
    }
}

final class SharpAIVoxtralRealtimeDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIVoxtralRealtimeAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIVoxtralRealtimeMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    // UNVERIFIED: placement, key names and dimension are assumed. Only the feature name
    // "ada_rmsnorm" is in model-coverage.csv. A real checkpoint may fail to load here.
    @ModuleInfo(key: "ada_rms_norm_t_cond") var adaNorm: SharpAIVoxtralRealtimeAdaNorm

    init(_ config: SharpAIVoxtralRealtimeTextConfiguration, adaNormDimension: Int) {
        self._selfAttn.wrappedValue = SharpAIVoxtralRealtimeAttention(config)
        self._mlp.wrappedValue = SharpAIVoxtralRealtimeMLP(
            hidden: config.hiddenSize, intermediate: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._adaNorm.wrappedValue = SharpAIVoxtralRealtimeAdaNorm(
            hidden: config.hiddenSize, dimension: adaNormDimension)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        condition: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        h = h + mlp(adaNorm(postAttentionLayerNorm(h), condition: condition))
        return h
    }
}

/// Layers and the final norm. Token embedding happens in the caller, so audio can be added first.
final class SharpAIVoxtralRealtimeDecoder: Module {

    let slidingWindow: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIVoxtralRealtimeDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIVoxtralRealtimeTextConfiguration, adaNormDimension: Int) {
        self.slidingWindow = config.slidingWindow
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAIVoxtralRealtimeDecoderLayer(config, adaNormDimension: adaNormDimension)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    /// `embeddings` is (batch, tokens, hidden). Returns the normed hidden states.
    func callAsFunction(
        _ embeddings: MLXArray, condition: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first, windowSize: slidingWindow)
        for (index, layer) in layers.enumerated() {
            h = layer(h, condition: condition, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}

/// Wraps the decoder under the HF `language_model.model.` prefix.
final class SharpAIVoxtralRealtimeLanguageModel: Module {

    @ModuleInfo(key: "model") var model: SharpAIVoxtralRealtimeDecoder

    init(_ config: SharpAIVoxtralRealtimeTextConfiguration, adaNormDimension: Int) {
        self._model.wrappedValue = SharpAIVoxtralRealtimeDecoder(
            config, adaNormDimension: adaNormDimension)
        super.init()
    }
}

/// Sinusoidal embedding of the delay, as (1, 1, dimension): [cos(...), sin(...)].
/// UNVERIFIED: frequency base 10000 and the cos-then-sin order follow the usual layout.
enum SharpAIVoxtralRealtimeDelay {
    static func embedding(_ delay: Int, dimension: Int) -> MLXArray {
        let half = dimension / 2
        let values: [Float] = (0 ..< half).map { index in
            Float(delay) * exp(-log(Float(10_000)) * Float(index) / Float(half))
        }
        let angles = MLXArray(values, [half])
        return concatenated([cos(angles), sin(angles)], axis: -1).reshaped(1, 1, -1)
    }
}
