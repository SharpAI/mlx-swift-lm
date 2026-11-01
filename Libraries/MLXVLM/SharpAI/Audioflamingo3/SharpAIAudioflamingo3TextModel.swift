// SharpAI fork overlay: audioflamingo3 text decoder (Qwen2 layout). Not part of upstream.
// Qwen2's blocks are internal and cannot take input embeddings, so this decoder is written here.
// Weight names match Qwen2, so the checkpoint keys stay the same.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIAudioflamingo3DecoderAttention: Module {
    let heads: Int
    let kvHeads: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIAudioflamingo3TextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.hiddenSize / heads
        self.heads = heads
        self.kvHeads = kvHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: true)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: true)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
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
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return oProj(output)
    }
}

final class SharpAIAudioflamingo3DecoderMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class SharpAIAudioflamingo3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIAudioflamingo3DecoderAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIAudioflamingo3DecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIAudioflamingo3TextConfiguration) {
        _attention.wrappedValue = SharpAIAudioflamingo3DecoderAttention(config)
        _mlp.wrappedValue = SharpAIAudioflamingo3DecoderMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
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

/// The `model.*` part of the Qwen2 decoder. It takes embeddings, so audio can be spliced in.
final class SharpAIAudioflamingo3DecoderModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIAudioflamingo3DecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIAudioflamingo3TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIAudioflamingo3DecoderLayer(config)
        }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
        }
        return norm(h)
    }
}

/// Qwen2 causal LM over embeddings: `model.*` plus an optional untied `lm_head`.
final class SharpAIAudioflamingo3LanguageModel: Module {
    @ModuleInfo(key: "model") var model: SharpAIAudioflamingo3DecoderModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: SharpAIAudioflamingo3TextConfiguration) {
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _model.wrappedValue = SharpAIAudioflamingo3DecoderModel(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    func embed(_ tokens: MLXArray) -> MLXArray {
        model.embedTokens(tokens)
    }

    func logits(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(embeddings, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
