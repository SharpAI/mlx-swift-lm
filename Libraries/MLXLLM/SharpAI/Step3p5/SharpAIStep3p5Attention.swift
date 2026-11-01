//
//  SharpAIStep3p5Attention.swift
//  MLXLLM
//
//  Self-attention for step3p5. Query head count varies per layer. Sliding layers use the
//  attention window, and full layers use none. QK-norm runs before RoPE. RoPE covers
//  head_dim * partial_rotary_factor channels, and some layers skip it via use_rope_layers.
//  A head-wise sigmoid gate (g_proj) scales each head's output when use_head_wise_attn_gate is set.
//  UNVERIFIED: the gate and RoPE layout come from vLLM step3p5.py, not from a run.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIStep3p5Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let useRope: Bool
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIStep3p5RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIStep3p5RMSNorm

    init(_ config: SharpAIStep3p5Configuration, layer: Int) {
        let heads = config.queryHeads(layer: layer)
        let kvHeads = config.attentionGroups
        let headDim = config.headDim

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)

        if config.useHeadWiseAttnGate {
            _gProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        } else {
            _gProj.wrappedValue = nil
        }

        _qNorm.wrappedValue = SharpAIStep3p5RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIStep3p5RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        self.useRope = config.useRopeLayers.isEmpty ? true : config.useRopeLayers[layer]
        self.rope = initializeRope(
            dims: config.rotaryDimensions(layer: layer),
            base: config.ropeBase(layer: layer),
            traditional: false,
            scalingConfig: config.ropeParameters(layer: layer),
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        let valuesHeadFirst = values.transposed(0, 2, 1, 3)

        if useRope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        // [B, L, heads, headDim] before the gate and the head merge.
        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: valuesHeadFirst,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)

        if let gProj {
            let gate = sigmoid(gProj(x)).asType(output.dtype)
            output = output * gate[.ellipsis, .newAxis]
        }

        return oProj(output.reshaped(B, L, -1))
    }
}
