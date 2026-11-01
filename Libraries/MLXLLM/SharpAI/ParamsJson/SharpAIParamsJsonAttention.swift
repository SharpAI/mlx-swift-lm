// SharpAI fork overlay: MLA attention for the ParamsJson EAGLE drafter. Not part of upstream.
// Structure copied from Mistral4Attention. The only addition is the llama 4 query scale.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIParamsJsonAttention: Module {
    let numHeads: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let scale: Float

    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIParamsJsonConfiguration) {
        numHeads = config.nHeads
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        qkNopeHeadDim = config.qkNopeHeadDim
        qHeadDim = config.qHeadDim
        scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(config.dim, config.qLoraRank, bias: false)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.normEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank, numHeads * qHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            config.dim, kvLoraRank + qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.normEps)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, config.dim, bias: false)

        rope = initializeRope(
            dims: qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        attnScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = qBProj(qALayerNorm(qAProj(x)))
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let (qNope, qPe) = (splitQ[0], splitQ[1])

        let raw = kvAProjWithMqa(x)
        let splitRaw = split(raw, indices: [kvLoraRank], axis: -1)
        let compressedKv = splitRaw[0]
        var kPe = splitRaw[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let (kNope, values) = (splitKv[0], splitKv[1])

        let offset = cache?.offset ?? 0
        let rotQPe = rope(qPe, offset: offset)
        let rotKPe = rope(kPe, offset: offset)
        let kPeExpanded = repeated(rotKPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPeExpanded], axis: -1)
        var queries = concatenated([qNope, rotQPe], axis: -1)
        queries = queries * attnScale

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}
