// SharpAI fork overlay: flint attention with a learned sink logit per head. Not part of upstream.
//
// Sinks use the `sinks:` argument of MLXFast.scaledDotProductAttention, as GPT-OSS does.
// Layers listed in no_rope_layer_indices skip RoPE. Ref: vllm/models/flint/model.py.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIFlintAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    // Key `self_attn.softmax_offset`, taken from the vLLM GGUF name map.
    // UNVERIFIED: the safetensors key and dtype. vLLM keeps this in fp32; here it uses the default dtype.
    @ParameterInfo(key: "softmax_offset") var softmaxOffset: MLXArray

    let rope: RoPE?

    init(_ config: SharpAIFlintConfiguration, usesRope: Bool) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * config.headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * config.headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _softmaxOffset.wrappedValue = zeros([config.attentionHeads])

        if usesRope {
            self.rope = RoPE(
                dimensions: config.headDim, traditional: false, base: config.resolvedRopeTheta)
        } else {
            self.rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, numHeads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, numKVHeads, -1)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = sharpAIFlintSinkAttention(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: softmaxOffset
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

/// Cache update and sink attention. Quantized caches have no sink path, so they are rejected.
private func sharpAIFlintSinkAttention(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray
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

    precondition(
        !(cache is QuantizedKVCacheProtocol),
        "Flint attention sinks do not support quantized KV caches.")
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
