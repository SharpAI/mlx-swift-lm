// SharpAI fork overlay: cohere_asr decoder (transformers CohereAsrDecoder). Not part of
// upstream mlx-swift-lm.
//
// Each layer is pre-norm: causal self-attention (KV cached), cross-attention to the encoder,
// then an MLP. Cross-attention keys and values are recomputed on every call, so there is no
// cross-attention cache yet.
// UNVERIFIED: no checkpoint was run. Layout follows modeling_cohere_asr.py in transformers.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Causal self-attention with a KV cache. Positions come from the learned table, not RoPE.
final class SharpAICohereAsrSelfAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAICohereAsrConfiguration) {
        let heads = config.attentionHeads
        let headDim = config.headDim
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        let width = heads * headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(width, config.hiddenSize, bias: config.attentionBias)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)

        let out = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

/// Attention from decoder states to the encoder output. No mask and no cache.
final class SharpAICohereAsrCrossAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAICohereAsrConfiguration) {
        let heads = config.attentionHeads
        let headDim = config.headDim
        self.heads = heads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        let width = heads * headDim
        self._qProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(config.hiddenSize, width, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(width, config.hiddenSize, bias: config.attentionBias)
        super.init()
    }

    /// `x` is (batch, tokens, hidden). `encoded` is (batch, frames, hidden), already projected.
    func callAsFunction(_ x: MLXArray, encoded: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        let s = encoded.dim(1)
        let q = qProj(x).reshaped(b, t, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(encoded).reshaped(b, s, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(encoded).reshaped(b, s, heads, headDim).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale,
            mask: MLXFast.ScaledDotProductAttentionMaskMode.none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, t, -1))
    }
}

/// Two-layer MLP with the decoder activation (ReLU in the released config).
final class SharpAICohereAsrDecoderMLP: Module {

    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let activation: (MLXArray) -> MLXArray

    init(_ config: SharpAICohereAsrConfiguration, activation: @escaping (MLXArray) -> MLXArray) {
        self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(activation(fc1(x)))
    }
}

final class SharpAICohereAsrDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var selfAttn: SharpAICohereAsrSelfAttention
    @ModuleInfo(key: "encoder_attn") var encoderAttn: SharpAICohereAsrCrossAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAICohereAsrDecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm
    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    init(_ config: SharpAICohereAsrConfiguration, activation: @escaping (MLXArray) -> MLXArray) {
        self._selfAttn.wrappedValue = SharpAICohereAsrSelfAttention(config)
        self._encoderAttn.wrappedValue = SharpAICohereAsrCrossAttention(config)
        self._mlp.wrappedValue = SharpAICohereAsrDecoderMLP(config, activation: activation)
        self._inputLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        self._postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: 1e-5)
        self._finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        encoded: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        h = h + encoderAttn(postAttentionLayerNorm(h), encoded: encoded)
        h = h + mlp(finalLayerNorm(h))
        return h
    }
}

/// Token and learned position embeddings, then the layers and a final norm.
final class SharpAICohereAsrDecoder: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "pos_emb") var posEmb: Embedding
    @ModuleInfo(key: "embedding_layernorm") var embeddingLayerNorm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "layers") var layers: [SharpAICohereAsrDecoderLayer]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ config: SharpAICohereAsrConfiguration) throws {
        let activation = try SharpAICohereAsrActivation.resolve(config.hiddenAct)
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._posEmb.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize)
        self._embeddingLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: 1e-5)
        self._proj.wrappedValue = Linear(
            config.encoderConfiguration.hiddenSize, config.hiddenSize, bias: true)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAICohereAsrDecoderLayer(config, activation: activation)
        }
        self._norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
        super.init()
    }

    /// `tokens` is (batch, tokens). `encoderStates` is the raw encoder output (batch, frames, enc).
    func callAsFunction(
        _ tokens: MLXArray, encoderStates: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        let encoded = proj(encoderStates)

        // Positions continue from the cache, so a step sees the same index as a full pass.
        let offset = cache?.first?.offset ?? 0
        let length = tokens.dim(1)
        let positions = MLXArray((0 ..< length).map { Int32($0 + offset) })

        var h = embeddingLayerNorm(embedTokens(tokens) + posEmb(positions))
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (index, layer) in layers.enumerated() {
            h = layer(h, encoded: encoded, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}
