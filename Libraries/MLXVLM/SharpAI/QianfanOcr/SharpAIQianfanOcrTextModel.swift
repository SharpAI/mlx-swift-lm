// SharpAI fork overlay: qianfan_ocr text decoder (Qwen3 layout). Not part of upstream mlx-swift-lm.
// The upstream Qwen3 blocks take token ids only, so this decoder is written here to accept
// embeddings with image features spliced in. Weight names match Qwen3 under `language_model.model`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIQianfanOcrDecoderAttention: Module {
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

    init(_ config: SharpAIQianfanOcrTextConfiguration) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim
        self.heads = heads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1)
        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

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

final class SharpAIQianfanOcrDecoderMLP: Module {
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

final class SharpAIQianfanOcrDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIQianfanOcrDecoderAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIQianfanOcrDecoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIQianfanOcrTextConfiguration) {
        _attention.wrappedValue = SharpAIQianfanOcrDecoderAttention(config)
        _mlp.wrappedValue = SharpAIQianfanOcrDecoderMLP(
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

/// The `model.*` part of the Qwen3 decoder. It takes embeddings, so image features can be spliced in.
final class SharpAIQianfanOcrDecoderModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIQianfanOcrDecoderLayer]
    let norm: RMSNorm

    init(_ config: SharpAIQianfanOcrTextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIQianfanOcrDecoderLayer(config)
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

/// Qwen3 causal LM over embeddings: `model.*` plus an optional untied `lm_head`.
final class SharpAIQianfanOcrLanguageModel: Module {
    @ModuleInfo(key: "model") var model: SharpAIQianfanOcrDecoderModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let kvHeads: [Int]

    init(_ config: SharpAIQianfanOcrTextConfiguration) {
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _model.wrappedValue = SharpAIQianfanOcrDecoderModel(config)
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
