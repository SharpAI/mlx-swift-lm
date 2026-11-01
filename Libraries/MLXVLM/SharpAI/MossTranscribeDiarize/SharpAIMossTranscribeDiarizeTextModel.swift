// SharpAI fork overlay: moss_transcribe_diarize text decoder (Qwen3 layout). Not part of upstream.
// Written here because the upstream Qwen3 blocks cannot take input embeddings.
// Weight names follow Qwen3 (`model.embed_tokens`, `self_attn.q_norm`, `lm_head`).
// UNVERIFIED: the checkpoint is assumed to be a Qwen3 decoder. The spec tags it "qwen3" only.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIMossTranscribeDiarizeDecoderAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIMossTranscribeDiarizeTextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        // Per-head RMSNorm acts on the last axis, so it runs before the head transpose.
        var queries = qNorm(qProj(x).reshaped(batch, length, heads, headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

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

final class SharpAIMossTranscribeDiarizeDecoderMLP: Module {
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

final class SharpAIMossTranscribeDiarizeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIMossTranscribeDiarizeDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIMossTranscribeDiarizeDecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIMossTranscribeDiarizeTextConfiguration) {
        _attention.wrappedValue = SharpAIMossTranscribeDiarizeDecoderAttention(config)
        _mlp.wrappedValue = SharpAIMossTranscribeDiarizeDecoderMLP(
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

/// The `model.*` part of the Qwen3 decoder. It takes embeddings, so audio can be spliced in.
final class SharpAIMossTranscribeDiarizeDecoderModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIMossTranscribeDiarizeDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIMossTranscribeDiarizeTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIMossTranscribeDiarizeDecoderLayer(config)
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

/// Causal LM over embeddings: `model.*` plus an optional untied `lm_head`.
final class SharpAIMossTranscribeDiarizeLanguageModel: Module {
    @ModuleInfo(key: "model") var model: SharpAIMossTranscribeDiarizeDecoderModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: SharpAIMossTranscribeDiarizeTextConfiguration) {
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _model.wrappedValue = SharpAIMossTranscribeDiarizeDecoderModel(config)
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
