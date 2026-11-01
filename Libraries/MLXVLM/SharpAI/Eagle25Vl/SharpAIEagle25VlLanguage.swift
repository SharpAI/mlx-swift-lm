// SharpAI fork overlay: Qwen2 text decoder for eagle_2_5_vl. Not part of upstream mlx-swift-lm.
//
// Same layout as Qwen2.swift (biased q/k/v, bias-free o and MLP, RMSNorm, RoPE). The repo's Qwen2
// types are internal or take token ids only, so this decoder accepts input embeddings, which
// the image features need. Keys stay under `language_model.model.*` and `language_model.lm_head.*`.
// UNVERIFIED: the text backbone is assumed to be plain Qwen2 with no sliding window.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIEagle25VlTextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIEagle25VlTextConfiguration) {
        let headDim = config.hiddenSize / config.attentionHeads
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
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

final class SharpAIEagle25VlTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ config: SharpAIEagle25VlTextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIEagle25VlTextLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIEagle25VlTextAttention
    let mlp: SharpAIEagle25VlTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIEagle25VlTextConfiguration) {
        _attention.wrappedValue = SharpAIEagle25VlTextAttention(config)
        self.mlp = SharpAIEagle25VlTextMLP(config)
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

/// Decoder stack. Takes embeddings, not token ids.
final class SharpAIEagle25VlTextBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIEagle25VlTextLayer]
    let norm: RMSNorm

    init(_ config: SharpAIEagle25VlTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIEagle25VlTextLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

/// Language model with the LM head. Tied models reuse `embed_tokens` as the head.
final class SharpAIEagle25VlTextLanguageModel: Module {
    @ModuleInfo var model: SharpAIEagle25VlTextBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: SharpAIEagle25VlTextConfiguration) {
        _model.wrappedValue = SharpAIEagle25VlTextBackbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(embeddings: embeddings, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
