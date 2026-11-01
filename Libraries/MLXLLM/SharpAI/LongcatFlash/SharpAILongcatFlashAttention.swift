// SharpAI fork overlay: longcat_flash MLA attention and dense MLP. Not part of upstream.
//
// Keeps kv_b_proj as in the checkpoint and splits it per call, instead of the absorbed
// embed_q / unembed_out form in mlx-lm. The two are equal because kv_b_proj is linear
// with no bias, so the mla_scale_kv_lora factor can be applied to its output.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAILongcatFlashAttention: Module {
    let numHeads: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float
    let qScale: Float?
    let kvScale: Float?
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAILongcatFlashConfiguration) {
        let hidden = config.hiddenSize
        let heads = config.numAttentionHeads
        let nope = config.qkNopeHeadDim
        let ropeDims = config.qkRopeHeadDim
        let valueDims = config.vHeadDim
        let qkDims = nope + ropeDims
        let qLora = config.qLoraRank
        let kvLora = config.kvLoraRank

        self.numHeads = heads
        self.qkRopeHeadDim = ropeDims
        self.qkNopeHeadDim = nope
        self.qHeadDim = qkDims
        self.vHeadDim = valueDims
        self.kvLoraRank = kvLora

        var scale = pow(Float(qkDims), -0.5)
        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0
            if mScaleAllDim != 0 {
                let factor = ropeScaling["factor"]?.asFloat() ?? 1
                if factor > 1 {
                    let s = 0.1 * mScaleAllDim * log(factor) + 1
                    scale = scale * s * s
                }
            }
        }
        self.scale = scale

        self.qScale =
            config.mlaScaleQLora
            ? Float(sqrt(Double(hidden) / Double(qLora))) : nil
        self.kvScale =
            config.mlaScaleKvLora
            ? Float(sqrt(Double(hidden) / Double(kvLora))) : nil

        self.rope = initializeRope(
            dims: ropeDims, base: config.ropeTheta, traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        self._qAProj.wrappedValue = Linear(hidden, qLora, bias: config.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLora)
        self._qBProj.wrappedValue = Linear(qLora, heads * qkDims, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(
            hidden, kvLora + ropeDims, bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLora)
        self._kvBProj.wrappedValue = Linear(kvLora, heads * (nope + valueDims), bias: false)
        self._oProj.wrappedValue = Linear(
            heads * valueDims, hidden, bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qBProj(qALayerNorm(qAProj(x)))
        if let qScale {
            q = q * qScale
        }
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        let compressed = kvAProjWithMqa(x)
        let splitCompressed = split(compressed, indices: [kvLoraRank], axis: -1)
        var kPe = splitCompressed[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(splitCompressed[0]))
        if let kvScale {
            kv = kv * kvScale
        }
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKv[0]
        let values = splitKv[1]

        let offset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: offset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: offset)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let keys = concatenated([kNope, kPe], axis: -1)
        let queries = concatenated([qNope, qPe], axis: -1)

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

/// Dense SwiGLU MLP. Each decoder layer holds two of these.
class SharpAILongcatFlashDenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
