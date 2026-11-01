//
//  SharpAIHyV3Attention.swift
//  MLXLLM
//
//  Grouped-query attention for hy_v3: per-head RMSNorm on Q and K (qk_norm),
//  then full-width RoPE.
//  Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/hy_v3/modeling_hy_v3.py
//  (HYV3Attention.__init__: q_proj, k_proj, v_proj, o_proj, q_norm, k_norm).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIHyV3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPELayer

    init(_ config: SharpAIHyV3Configuration) {
        let headDim = config.headDim
        self.attentionHeads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize, config.attentionHeads * headDim, bias: config.attentionBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * headDim, bias: config.attentionBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, config.kvHeads * headDim, bias: config.attentionBias)
        _wo.wrappedValue = Linear(config.attentionHeads * headDim, config.hiddenSize, bias: false)

        if config.qkNorm {
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        }

        self.rope = initializeRope(
            dims: headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(B, L, attentionHeads, -1)
        keys = keys.reshaped(B, L, kvHeads, -1)

        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

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
