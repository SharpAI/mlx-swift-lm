//
//  SharpAIExaoneMoeAttention.swift
//  MLXLLM
//
//  Self-attention for exaone_moe. QK-norm runs on the head dim before RoPE.
//  RoPE is full-width, neox style. Full-attention layers are NoPE when sliding
//  layers exist (vLLM use_rope).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIExaoneMoeAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let slidingWindow: Int?
    let rope: RoPELayer?

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAIExaoneMoeConfiguration, layer: Int, mtp: Bool) {
        let heads = config.attentionHeads
        let kvHeads = config.kvHeads
        let headDim = config.headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)
        self.slidingWindow = config.windowSize(layer: layer, mtp: mtp)

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

        if config.usesRope(layer: layer, mtp: mtp) {
            self.rope = initializeRope(
                dims: headDim, base: config.ropeTheta,
                traditional: false, scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

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

        return oProj(output)
    }
}
