//
//  SharpAIMimoV2Attention.swift
//  MLXLLM
//
//  Attention for mimo_v2. Sliding-window layers use the swa_* geometry, global
//  layers the base geometry. v is scaled by attention_value_scale before SDPA.
//  Sinks exist only on sliding-window layers, as in vLLM. Q/K/V are separate
//  Linear modules; sanitize() splits the checkpoint's fused qkv_proj.
//  Verified against: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/modeling_mimo_v2.py
//  (MiMoV2Attention defines an optional attention_sink_bias on the SWA path; the key name matches).
//  UNVERIFIED: MLX SDPA with v_head_dim != head_dim.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Sink-aware cache update and attention. Quantized caches have no sink path.
private func sharpAIMimoV2SinkAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }

    if let attentionCache = cache as? KVCacheAttentionProtocol, sinks == nil {
        return attentionCache.updateAndAttend(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask)
    } else if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask,
            sinks: sinks
        )
    }
}

class SharpAIMimoV2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let vHeadDim: Int
    let scale: Float
    let valueScale: Float?
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ParameterInfo(key: "attention_sink_bias") var attentionSinkBias: MLXArray?

    init(_ config: SharpAIMimoV2Configuration, isSlidingWindow: Bool) {
        let heads = isSlidingWindow ? config.swaAttentionHeads : config.attentionHeads
        let kvHeads = isSlidingWindow ? config.swaKvHeads : config.kvHeads
        let dim = isSlidingWindow ? config.swaHeadDim : config.headDim
        let vDim = isSlidingWindow ? config.swaVHeadDim : config.vHeadDim
        let hiddenSize = config.hiddenSize

        self.numHeads = heads
        self.numKVHeads = kvHeads
        self.headDim = dim
        self.vHeadDim = vDim
        self.scale = pow(Float(dim), -0.5)
        self.valueScale = config.attentionValueScale

        _wq.wrappedValue = Linear(hiddenSize, heads * dim, bias: config.attentionBias)
        _wk.wrappedValue = Linear(hiddenSize, kvHeads * dim, bias: config.attentionBias)
        _wv.wrappedValue = Linear(hiddenSize, kvHeads * vDim, bias: config.attentionBias)
        _wo.wrappedValue = Linear(heads * vDim, hiddenSize, bias: false)

        let hasSinks = isSlidingWindow && config.addSwaAttentionSinkBias
        let sinks: MLXArray? = hasSinks ? zeros([heads]) : nil
        _attentionSinkBias.wrappedValue = sinks

        let rotaryDims = Int(config.partialRotaryFactor * Float(dim))
        let ropeTheta = isSlidingWindow ? config.swaRopeTheta : config.ropeTheta
        self.rope = RoPE(dimensions: rotaryDims, traditional: false, base: ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = wq(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(B, L, numKVHeads, vHeadDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)

        if let valueScale {
            v = v * valueScale
        }

        let output = sharpAIMimoV2SinkAttention(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: attentionSinkBias
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}
