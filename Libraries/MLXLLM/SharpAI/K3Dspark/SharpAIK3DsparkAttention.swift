// SharpAI fork overlay: k3_dspark MLA attention, MLP and decoder layer. Not part of upstream.
//
// Non-absorbed MLA with no KV cache. Draft queries attend without a causal mask to the
// draft keys and to the target-derived context keys. The context keys come from the same
// per-layer kv_a_proj_with_mqa and kv_b_proj as the draft keys, as in the vLLM draft.
// UNVERIFIED: the draft and context position convention. Only contiguous offsets are used.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

class SharpAIK3DsparkAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIK3DsparkConfiguration) {
        numHeads = config.numAttentionHeads
        qkNopeHeadDim = config.qkNopeHeadDim
        qkRopeHeadDim = config.qkRopeHeadDim
        vHeadDim = config.vHeadDim
        kvLoraRank = config.kvLoraRank

        // YaRN softmax correction, as in vLLM's K3 MLA: only when the factor exceeds 1.
        var softmaxScale = pow(Float(config.qkNopeHeadDim + config.qkRopeHeadDim), -0.5)
        if let scaling = config.ropeScaling,
            scaling["attention_factor"]?.asFloat() != 1.0,
            let factor = scaling["factor"]?.asFloat(), factor > 1
        {
            let mscaleAllDim = scaling["mscale_all_dim"]?.asFloat() ?? 0
            let mscale = 0.1 * mscaleAllDim * log(factor) + 1.0
            softmaxScale *= mscale * mscale
        }
        scale = softmaxScale

        rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)

        guard let qLoraRank = config.qLoraRank else {
            preconditionFailure("k3_dspark requires q_lora_rank. Validate the configuration first.")
        }
        let qkHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self._qAProj.wrappedValue = Linear(config.hiddenSize, qLoraRank, bias: false)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: config.rmsNormEps)
        self._qBProj.wrappedValue = Linear(
            qLoraRank, config.numAttentionHeads * qkHeadDim, bias: false)
        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, config.kvLoraRank + config.qkRopeHeadDim, bias: false)
        self._kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: config.kvLoraRank, eps: config.rmsNormEps)
        self._kvBProj.wrappedValue = Linear(
            config.kvLoraRank,
            config.numAttentionHeads * (config.qkNopeHeadDim + config.vHeadDim),
            bias: false)
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.vHeadDim, config.hiddenSize, bias: false)
    }

    /// Latent KV for one sequence. Returns key nope [B, H, L, nope], values [B, H, L, v],
    /// and the shared rope key [B, 1, L, rope].
    private func latentKV(_ states: MLXArray, offset: Int) -> (MLXArray, MLXArray, MLXArray) {
        let (B, L) = (states.dim(0), states.dim(1))
        let compressed = split(kvAProjWithMqa(states), indices: [kvLoraRank], axis: -1)
        let kPe = compressed[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        let kv = kvBProj(kvALayerNorm(compressed[0]))
            .reshaped(B, L, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kvSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kRope = applyRotaryPosition(rope, to: kPe, offset: .scalar(offset))
        return (kvSplit[0], kvSplit[1], kRope)
    }

    /// `x` is the normed draft hidden state [B, L, hidden]. `context` is the combined
    /// target state [B, Lc, hidden], which is not normed again.
    func callAsFunction(
        _ x: MLXArray, context: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkHeadDim = qkNopeHeadDim + qkRopeHeadDim

        let q = qBProj(qALayerNorm(qAProj(x)))
            .reshaped(B, L, numHeads, qkHeadDim)
            .transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qSplit[0]
        let qPe = applyRotaryPosition(rope, to: qSplit[1], offset: .scalar(draftOffset))

        var (keyNope, values, keyPe) = latentKV(x, offset: draftOffset)
        if let context {
            let (ctxNope, ctxValues, ctxPe) = latentKV(context, offset: contextOffset)
            keyNope = concatenated([ctxNope, keyNope], axis: 2)
            values = concatenated([ctxValues, values], axis: 2)
            keyPe = concatenated([ctxPe, keyPe], axis: 2)
        }
        keyPe = repeated(keyPe, count: numHeads, axis: 1)

        let queries = concatenated([qNope, qPe], axis: -1)
        let keys = concatenated([keyNope, keyPe], axis: -1)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class SharpAIK3DsparkMLP: Module {
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

class SharpAIK3DsparkDecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIK3DsparkAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: SharpAIK3DsparkMLP

    init(_ config: SharpAIK3DsparkConfiguration) {
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._selfAttn.wrappedValue = SharpAIK3DsparkAttention(config)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._mlp.wrappedValue = SharpAIK3DsparkMLP(
            hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(
        _ x: MLXArray, context: MLXArray?, contextOffset: Int, draftOffset: Int
    ) -> MLXArray {
        var h =
            x
            + selfAttn(
                inputLayerNorm(x), context: context,
                contextOffset: contextOffset, draftOffset: draftOffset)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}
