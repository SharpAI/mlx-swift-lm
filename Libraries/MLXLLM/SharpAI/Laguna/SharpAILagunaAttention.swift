//
//  SharpAILagunaAttention.swift
//  MLXLLM
//
//  Self-attention for laguna. Head count varies per layer. QK-norm runs before RoPE.
//  RoPE covers only head_dim * partial_rotary_factor channels. An optional softplus
//  gate (g_proj) scales the attention output per head or per element.
//  UNVERIFIED: the gate and RoPE layout come from vLLM laguna.py, not from a run.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILagunaAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatePerHead: Bool
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: SharpAILagunaConfiguration, layer: Int) {
        let heads = config.attentionHeads(layer: layer)
        let kvHeads = config.kvHeads
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

        switch config.gating {
        case .perHead:
            self.gatePerHead = true
            _gProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        case .perElement:
            self.gatePerHead = false
            _gProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        case .off, .unknown:
            self.gatePerHead = false
            _gProj.wrappedValue = nil
        }

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        // Validation has already checked that every rope dict carries rope_theta.
        precondition(
            config.ropeTheta(layer: layer) != nil,
            "laguna: rope_theta missing. Validate the configuration first.")
        self.rope = initializeRope(
            dims: config.rotaryDimensions(layer: layer),
            base: config.ropeTheta(layer: layer) ?? 10000,
            traditional: false,
            scalingConfig: config.ropeScaling(layer: layer),
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

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

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
            let gate = softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                output = output * gate[.ellipsis, .newAxis]
            } else {
                output = output * gate.reshaped(B, L, numHeads, headDim)
            }
        }

        return oProj(output.reshaped(B, L, -1))
    }
}
