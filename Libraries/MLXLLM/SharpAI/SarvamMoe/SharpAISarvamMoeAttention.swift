//
//  SharpAISarvamMoeAttention.swift
//  MLXLLM
//
//  GQA attention with optional per-head QK RMSNorm and partial RoPE. Matches
//  modeling_sarvam_moe.py: fused `query_key_value` splits into (q, k, v) on the head axis,
//  QK norm runs before RoPE, and the output projection is `dense`.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAISarvamMoeAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var wo: Linear
    @ModuleInfo(key: "query_layernorm") var qNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var kNorm: RMSNorm?

    init(_ config: SharpAISarvamMoeConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = initializeRope(
            dims: config.ropeDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        _qkv.wrappedValue = Linear(
            config.hiddenSize,
            (config.attentionHeads + 2 * config.kvHeads) * config.headDim,
            bias: config.useQKVBias
        )
        _wo.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: config.useBias)

        if config.useQKNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        } else {
            _qNorm.wrappedValue = nil
            _kNorm.wrappedValue = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qSize = heads * headDim
        let kSize = kvHeads * headDim
        let parts = split(qkv(x), indices: [qSize, qSize + kSize], axis: -1)

        var queries = parts[0].reshaped(B, L, heads, headDim)
        var keys = parts[1].reshaped(B, L, kvHeads, headDim)
        var values = parts[2].reshaped(B, L, kvHeads, headDim)

        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

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
        .reshaped(B, L, -1)

        return wo(output)
    }
}
