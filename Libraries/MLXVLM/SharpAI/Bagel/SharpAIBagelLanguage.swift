// SharpAI fork overlay: Qwen2 text decoder with QK norm, used by the bagel model. Not part of
// upstream mlx-swift-lm.
//
// Port of the Qwen2 block in Libraries/MLXLLM/Models/Qwen2.swift, with the q_norm and k_norm
// pair from Qwen3.swift. Kept inside MLXVLM because the MLXLLM Qwen2 types are not visible here.
// Generation-expert (`_moe_gen`) weights are not modeled.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIBagelAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ config: SharpAIBagelConfiguration.TextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.hiddenSize / heads

        numHeads = heads
        numKVHeads = kvHeads
        scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: true)
        _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        rope = RoPE(
            dimensions: headDim, traditional: config.ropeTraditional, base: config.ropeTheta,
            scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, numHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, numKVHeads, -1)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

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
        .reshaped(B, L, -1)

        return wo(output)
    }
}

final class SharpAIBagelMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIBagelDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIBagelAttention
    @ModuleInfo var mlp: SharpAIBagelMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIBagelConfiguration.TextConfiguration) {
        _attention.wrappedValue = SharpAIBagelAttention(config)
        mlp = SharpAIBagelMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
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

final class SharpAIBagelTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [SharpAIBagelDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIBagelConfiguration.TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIBagelDecoderLayer(config)
        }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
}

/// Text decoder over input embeddings. Weight keys: `model.*` and `lm_head`, under the
/// `language_model` prefix in the bagel checkpoint.
final class SharpAIBagelLanguageModel: Module {
    let model: SharpAIBagelTextModel
    let tieWordEmbeddings: Bool

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIBagelConfiguration.TextConfiguration) {
        model = SharpAIBagelTextModel(config)
        tieWordEmbeddings = config.tieWordEmbeddings
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in model.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        let out = model.norm(h)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }
}
