// SharpAI fork overlay: Llama-2 decoder for OpenVLA. Not part of upstream mlx-swift-lm.
//
// Copied from MLXLLM's Llama, because the upstream layers are internal and the
// decoder must accept input embeddings. Module paths match HF Llama:
// `model.embed_tokens`, `model.layers.N.*`, `model.norm`, `lm_head`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIOpenvlaAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIOpenvlaConfiguration.TextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.resolvedHeadDim
        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.rope = initializeRope(
            dims: headDim, base: config.ropeTheta, traditional: config.ropeTraditional,
            scalingConfig: nil, maxPositionEmbeddings: nil)

        self._qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

final class SharpAIOpenvlaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: SharpAIOpenvlaConfiguration.TextConfiguration) {
        self._gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
        self._up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIOpenvlaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIOpenvlaAttention
    @ModuleInfo var mlp: SharpAIOpenvlaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIOpenvlaConfiguration.TextConfiguration) {
        self._attention.wrappedValue = SharpAIOpenvlaAttention(config)
        self._mlp.wrappedValue = SharpAIOpenvlaMLP(config)
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Llama body (`model.*`). Takes embeddings, so image features can be merged before the layers.
final class SharpAIOpenvlaLlamaBody: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [SharpAIOpenvlaDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: SharpAIOpenvlaConfiguration.TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIOpenvlaDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

/// Llama text model with an LM head. Module paths: `model.*` and `lm_head`.
final class SharpAIOpenvlaLanguageModel: Module {
    let kvHeads: [Int]

    @ModuleInfo var model: SharpAIOpenvlaLlamaBody
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIOpenvlaConfiguration.TextConfiguration) {
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self._model.wrappedValue = SharpAIOpenvlaLlamaBody(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    func logits(_ hidden: MLXArray) -> MLXArray {
        lmHead(hidden)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model(embeddings: embed(tokens), cache: cache))
    }
}
