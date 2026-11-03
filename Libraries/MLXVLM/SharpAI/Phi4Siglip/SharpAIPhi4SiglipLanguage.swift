// SharpAI fork overlay: Phi-3-style text decoder for phi4-siglip. Not part of upstream.
//
// Same layout as Phi3.swift: fused `qkv_proj` and `gate_up_proj`, RMSNorm, partial RoPE. The
// upstream Phi3 types are internal and Phi3ModelInner takes token ids only, so this decoder
// takes input embeddings, which the image features need.
// UNVERIFIED: the text backbone is assumed to use the Phi-3 tensor layout.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class SharpAIPhi4SiglipAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIPhi4SiglipConfiguration) {
        let headDim = config.headDim
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        _qkvProj.wrappedValue = Linear(
            config.hiddenSize, (config.attentionHeads + 2 * config.kvHeads) * headDim, bias: false)
        _oProj.wrappedValue = Linear(config.attentionHeads * headDim, config.hiddenSize, bias: false)
        self.rope = RoPE(
            dimensions: config.ropeDimensions, traditional: false, base: config.ropeTheta,
            scale: 1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))

        let queryEnd = heads * headDim
        let qkv = split(qkvProj(x), indices: [queryEnd, queryEnd + kvHeads * headDim], axis: -1)
        var queries = qkv[0].reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)

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

final class SharpAIPhi4SiglipMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ config: SharpAIPhi4SiglipConfiguration) {
        _gateUp.wrappedValue = Linear(
            config.hiddenSize, 2 * config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(gateUp(x), parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

final class SharpAIPhi4SiglipLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: SharpAIPhi4SiglipAttention
    let mlp: SharpAIPhi4SiglipMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: SharpAIPhi4SiglipConfiguration) {
        _attention.wrappedValue = SharpAIPhi4SiglipAttention(config)
        self.mlp = SharpAIPhi4SiglipMLP(config)
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

/// Text decoder plus the vision path, under `model.*` as in LLaVA-style checkpoints.
/// The decoder takes embeddings. Token embedding happens in the model.
final class SharpAIPhi4SiglipBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIPhi4SiglipLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "vision_tower") var visionTower: SharpAIPhi4SiglipVisionTower
    /// `[Linear, GELU, Linear]` stored as an array so the checkpoint keys `mm_projector.0.*` and
    /// `mm_projector.2.*` unflatten into it (a dictionary-style Module with the keys "0" and "2"
    /// cannot take them: `ModuleParameters.unflattened` builds an array for numeric keys).
    @ModuleInfo(key: "mm_projector") var mmProjectorLayers: [Module]

    init(_ config: SharpAIPhi4SiglipConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in
            SharpAIPhi4SiglipLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _visionTower.wrappedValue = SharpAIPhi4SiglipVisionTower(config.visionConfiguration)
        _mmProjectorLayers.wrappedValue = [
            Linear(config.mmHiddenSize, config.hiddenSize, bias: true),
            GELU(),
            Linear(config.hiddenSize, config.hiddenSize, bias: true),
        ]
    }

    func mmProjector(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in mmProjectorLayers {
            h = (layer as! UnaryLayer)(h)
        }
        return h
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
