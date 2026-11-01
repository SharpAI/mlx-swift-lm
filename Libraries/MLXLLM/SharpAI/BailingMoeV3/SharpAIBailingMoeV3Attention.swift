//
//  SharpAIBailingMoeV3Attention.swift
//  MLXLLM
//
//  MLA attention for the full-attention layers of bailing_moe_v3. Keys and
//  values are expanded per head and cached as-is (no absorbed-weight path).
//  Head-wise sigmoid gate: g_proj gives one scalar per head, applied before dense.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIBailingMoeV3Attention: Module {
    let heads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float
    let rope: SharpAIBailingMoeV3RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ config: SharpAIBailingMoeV3Configuration) {
        let heads = config.attentionHeads
        let nope = config.qkNopeHeadDim
        let rope = config.qkRopeHeadDim
        let vDim = config.vHeadDim
        let latent = config.kvLoraRank

        self.heads = heads
        self.qkNopeHeadDim = nope
        self.qkRopeHeadDim = rope
        self.vHeadDim = vDim
        self.kvLoraRank = latent
        self.scale = pow(Float(nope + rope), -0.5)
        self.rope = SharpAIBailingMoeV3RoPE(
            dims: rope, base: config.ropeTheta, mropeSection: config.mropeSection)

        _qProj.wrappedValue = Linear(config.hiddenSize, heads * (nope + rope), bias: false)
        _kvAProjWithMQA.wrappedValue = Linear(config.hiddenSize, latent + rope, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: latent, eps: config.rmsNormEps)
        _kvBProj.wrappedValue = Linear(latent, heads * (nope + vDim), bias: false)

        if config.gatedAttentionProjGranularityType == "head_wise" {
            _gProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        } else {
            _gProj.wrappedValue = nil
        }
        _dense.wrappedValue = Linear(heads * vDim, config.hiddenSize, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let q = qProj(x).reshaped(B, L, heads, qkNopeHeadDim + qkRopeHeadDim)
            .transposed(0, 2, 1, 3)
        let qNope = q[.ellipsis, ..<qkNopeHeadDim]
        let qPe = q[.ellipsis, qkNopeHeadDim...]

        let latent = split(kvAProjWithMQA(x), indices: [kvLoraRank], axis: -1)
        let kvLatent = latent[0]
        let kPe = latent[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let kv = kvBProj(kvALayerNorm(kvLatent))
            .reshaped(B, L, heads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kNope = kv[.ellipsis, ..<qkNopeHeadDim]
        let values = kv[.ellipsis, qkNopeHeadDim...]

        let offset = cache?.ropeOffset
        let queries = concatenated([qNope, rope(qPe, offset: offset)], axis: -1)
        let kPeRotated = broadcast(rope(kPe, offset: offset), to: [B, heads, L, qkRopeHeadDim])
        let keys = concatenated([kNope, kPeRotated], axis: -1)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        output = output.transposed(0, 2, 1, 3)

        if let gProj {
            let gate = sigmoid(gProj(x).asType(.float32)).asType(x.dtype)
            output = output * gate.expandedDimensions(axis: -1)
        }

        return dense(output.reshaped(B, L, -1))
    }
}
