// SharpAI fork overlay: nemotron_parse mBART-style text decoder. Not part of upstream mlx-swift-lm.
//
// Reference (Verified against):
//   - https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-v1.1/resolve/main/hf_nemotron_parse_modeling.py
//     (`NemotronParseDecoder`: scaled word embedding, `layernorm_embedding`, pre-norm decoder layers,
//     final `layer_norm`; `MBartDecoderLayer` from transformers)
//   - https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nemotron_parse.py
//     (`MBartDecoderNoPos` and `MBartDecoderLayer`: pre-LN self-attn, cross-attn, FFN. The
//     checkpoint's `decoder.embed_positions.weight` is skipped there, and the reference adds no
//     positions, so this port adds none either.)
//
// Layer order (pre-LN, MBart): x += SelfAttn(LN_self(x)); x += CrossAttn(LN_enc(x), encoder);
// x += FC2(GELU(FC1(LN_final(x)))). Embeddings are scaled by sqrt(d_model) because
// `scale_embedding: true`. Activation is exact GELU (`activation_function: gelu`).
//
// Cache: self-attention keys and values go through `KVCache`, one entry per layer. Cross-attention
// keys and values are recomputed from the encoder states on each call (no cache for them).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAINemotronParseDecoderSelfAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, heads: Int) {
        self.heads = heads
        self.headDim = dim / heads
        self.scale = pow(Float(dim / heads), -0.5)
        _qProj.wrappedValue = Linear(dim, dim, bias: true)
        _kProj.wrappedValue = Linear(dim, dim, bias: true)
        _vProj.wrappedValue = Linear(dim, dim, bias: true)
        _outProj.wrappedValue = Linear(dim, dim, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return outProj(output)
    }
}

final class SharpAINemotronParseDecoderCrossAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dim: Int, heads: Int) {
        self.heads = heads
        self.headDim = dim / heads
        self.scale = pow(Float(dim / heads), -0.5)
        _qProj.wrappedValue = Linear(dim, dim, bias: true)
        _kProj.wrappedValue = Linear(dim, dim, bias: true)
        _vProj.wrappedValue = Linear(dim, dim, bias: true)
        _outProj.wrappedValue = Linear(dim, dim, bias: true)
    }

    /// `encoderStates` is `[batch, sourceLength, dim]`. Unmasked, as in the reference.
    func callAsFunction(_ x: MLXArray, encoderStates: MLXArray) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let sourceLength = encoderStates.dim(1)
        let queries = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
        let keys = kProj(encoderStates)
            .reshaped(batch, sourceLength, heads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(encoderStates)
            .reshaped(batch, sourceLength, heads, headDim).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return outProj(output)
    }
}

final class SharpAINemotronParseDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAINemotronParseDecoderSelfAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "encoder_attn") var encoderAttn: SharpAINemotronParseDecoderCrossAttention
    @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(dim: Int, heads: Int, ffnDim: Int) {
        _selfAttn.wrappedValue = SharpAINemotronParseDecoderSelfAttention(dim: dim, heads: heads)
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dim)
        _encoderAttn.wrappedValue = SharpAINemotronParseDecoderCrossAttention(dim: dim, heads: heads)
        _encoderAttnLayerNorm.wrappedValue = LayerNorm(dimensions: dim)
        _fc1.wrappedValue = Linear(dim, ffnDim, bias: true)
        _fc2.wrappedValue = Linear(ffnDim, dim, bias: true)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: dim)
    }

    func callAsFunction(
        _ x: MLXArray,
        encoderStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var h = x + selfAttn(selfAttnLayerNorm(x), mask: mask, cache: cache)
        h = h + encoderAttn(encoderAttnLayerNorm(h), encoderStates: encoderStates)
        h = h + fc2(gelu(fc1(finalLayerNorm(h))))
        return h
    }
}

/// `decoder.*`: the MBart decoder stack (no `lm_head`; that lives on the top-level model).
final class SharpAINemotronParseDecoder: Module {
    let embedScale: Float
    let layers: [SharpAINemotronParseDecoderLayer]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layernorm_embedding") var layernormEmbedding: LayerNorm
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(
        vocabularySize: Int, dim: Int, heads: Int, layers count: Int, ffnDim: Int
    ) {
        self.embedScale = sqrt(Float(dim))
        _embedTokens.wrappedValue = Embedding(embeddingCount: vocabularySize, dimensions: dim)
        _layernormEmbedding.wrappedValue = LayerNorm(dimensions: dim)
        self.layers = (0 ..< count).map { _ in
            SharpAINemotronParseDecoderLayer(dim: dim, heads: heads, ffnDim: ffnDim)
        }
        _layerNorm.wrappedValue = LayerNorm(dimensions: dim)
    }

    /// `tokens` is `[batch, length]` decoder input ids. `cache`, when given, has one entry per layer.
    func callAsFunction(
        _ tokens: MLXArray, encoderStates: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        var h = embedTokens(tokens) * embedScale
        h = layernormEmbedding(h)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (index, layer) in layers.enumerated() {
            h = layer(h, encoderStates: encoderStates, mask: mask, cache: cache?[index])
        }
        return layerNorm(h)
    }
}
