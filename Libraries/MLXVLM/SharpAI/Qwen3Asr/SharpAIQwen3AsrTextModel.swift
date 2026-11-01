// SharpAI fork overlay: qwen3_asr text decoder (a Qwen3 decoder). Not part of upstream
// mlx-swift-lm.
//
// Same layer layout as upstream Qwen3 (q_norm and k_norm, SwiGLU MLP). The upstream class
// only takes token ids, and this model needs to feed in merged audio and text embeddings, so
// the decoder is copied here. Upstream types are used only through public helpers.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIQwen3AsrAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIQwen3AsrTextConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: false)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: false)
        self._oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        self.rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(b, l, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(b, l, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(b, l, kvHeads, -1).transposed(0, 2, 1, 3)

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
        .reshaped(b, l, -1)

        return oProj(output)
    }
}

final class SharpAIQwen3AsrMLP: Module {

    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIQwen3AsrDecoderLayer: Module {

    @ModuleInfo(key: "self_attn") var attention: SharpAIQwen3AsrAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIQwen3AsrMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIQwen3AsrTextConfiguration) {
        self._attention.wrappedValue = SharpAIQwen3AsrAttention(config)
        self._mlp.wrappedValue = SharpAIQwen3AsrMLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Token embedding, decoder layers, and final norm. Takes embeddings, not token ids, so audio
/// embeddings can be merged in before the first layer.
public final class SharpAIQwen3AsrTextModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIQwen3AsrDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: SharpAIQwen3AsrTextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            SharpAIQwen3AsrDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        embedTokens(tokens)
    }

    /// `embeddings` is (batch, length, hidden).
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}
