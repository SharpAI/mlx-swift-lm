//
//  SharpAIParam2moeAttention.swift
//  MLXLLM
//
//  Grouped-query attention for param2moe. The checkpoint fuses Q, K and V into
//  `attention.query_key_value`, split as [Q | K | V]. Optional per-head RMS norms on Q and K
//  run before RoPE. Keys keep the HF names: query_key_value, dense, query_layernorm, key_layernorm.
//  UNVERIFIED: no checkpoint was run. The layout comes from vLLM param2moe.py.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIParam2moeAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm?

    init(_ config: SharpAIParam2moeConfiguration) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _queryKeyValue.wrappedValue = Linear(
            config.hiddenSize,
            (config.attentionHeads + 2 * config.kvHeads) * config.headDim,
            bias: config.useQKVBias
        )
        _dense.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: config.useBias)

        if config.useQKNorm {
            _queryLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.headDim, eps: config.rmsNormEps)
            _keyLayerNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        } else {
            _queryLayerNorm.wrappedValue = nil
            _keyLayerNorm.wrappedValue = nil
        }

        self.rope = initializeRope(
            dims: config.rotaryDimensions,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim
        let splits = split(queryKeyValue(x), indices: [qSize, qSize + kvSize], axis: -1)

        var queries = splits[0].reshaped(B, L, numHeads, headDim)
        var keys = splits[1].reshaped(B, L, numKVHeads, headDim)
        let values = splits[2].reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let queryLayerNorm { queries = queryLayerNorm(queries) }
        if let keyLayerNorm { keys = keyLayerNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

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

        return dense(output)
    }
}
