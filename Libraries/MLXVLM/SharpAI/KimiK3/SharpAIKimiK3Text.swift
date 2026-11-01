// SharpAI fork overlay: kimi_k3 text backbone (KimiLinearForCausalLM). Not part of upstream mlx-swift-lm.
//
// Port of moonshotai/Kimi-K3 modeling_kimi_linear.py (fetched 2026-10-30):
//   KimiLinearModel, KimiDecoderLayer (incl. _forward_attn_residual), KimiMLAAttention (NoPE, output gate),
//   KimiDeltaAttention (KDA, full-rank gate, lower-bound decay), KimiSparseMoeBlock (sigmoid router with
//   e_score_correction_bias, latent down/up projection, shared experts), _apply_attn_res.
// Kernels: the KDA recurrence is SharpAIOverlayGatedDeltaKDA (MLXLMCommon). The q/k L2 norm (eps 1e-6), the
// K^-0.5 query scale, the sigmoid beta and the gated RMS norm follow fla (chunk_kda, l2norm, fused_norm_gate).
//
// Scope: the text decoder only. Routed experts run through a stacked SwitchGLU (gather dispatch over the top-k
// experts); the sanitizer stacks the per-expert w1/w2/w3 tensors. The vision tower and the MXFP4-packed expert
// tensors of the released checkpoint are not mapped yet (see SharpAIKimiK3Sanitizer).

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Activation

/// `SituAndMul` from modeling_kimi_linear.py: beta * tanh(gate / beta) * sigmoid(gate) * up, where `up` is first
/// passed through linear_beta * tanh(up / linear_beta) when linear_beta is set. Plain SiLU gating for `silu`.
struct SharpAIKimiK3Activation: Sendable {
    let usesSitu: Bool
    let beta: Float
    let linearBeta: Float?

    init(_ config: SharpAIKimiK3TextConfiguration) {
        usesSitu = config.hiddenAct == "situ"
        // Mirrors `beta or 1.0` in the reference: a missing or zero beta means 1.
        if let value = config.activationSituBeta, value != 0 {
            beta = value
        } else {
            beta = 1
        }
        linearBeta = config.activationSituLinearBeta
    }

    func callAsFunction(gate: MLXArray, up: MLXArray) -> MLXArray {
        guard usesSitu else { return silu(gate) * up }
        let g = gate.asType(.float32)
        var u = up.asType(.float32)
        let betaArray = MLXArray(beta)
        let situA = betaArray * MLX.tanh(g / betaArray) * sigmoid(g)
        if let linearBeta {
            let linearArray = MLXArray(linearBeta)
            u = linearArray * MLX.tanh(u / linearArray)
        }
        return (situA * u).asType(gate.dtype)
    }
}

// MARK: - Norm weight holder

/// Holds the `weight` of a gated-free RMS norm whose eps lives in the caller. Used by the attention-residual
/// bank, which needs the raw weight (`norm.weight` in `_apply_attn_res`).
final class SharpAIKimiK3NormWeight: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ dimensions: Int) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }
}

/// `_apply_attn_res`: softmax over the block sources of (RMS-normalized source . (norm.weight * proj.weight)),
/// then a weighted sum of the raw sources. `sources` are `[B, S, H]`; `proj` is `[1, H]`.
func sharpAIKimiK3AttnRes(
    sources: [MLXArray], proj: MLXArray, norm: SharpAIKimiK3NormWeight, eps: Float
) -> MLXArray {
    let dtype = sources[0].dtype
    let values = MLX.stacked(sources, axis: -2).asType(.float32)  // [B, S, N, H]
    let rstd = MLX.rsqrt(MLX.mean(MLX.square(values), axis: -1, keepDims: true) + MLXArray(eps))
    let scoreWeight = (norm.weight * proj.reshaped(-1)).asType(.float32)  // [H]
    let logits = MLX.sum(values * rstd * scoreWeight, axis: -1)  // [B, S, N]
    let probs = MLX.softmax(logits, axis: -1, precise: true)
    let mixed = MLX.sum(MLX.expandedDimensions(probs, axis: -1) * values, axis: -2)
    return mixed.asType(dtype)
}

// MARK: - Token mixers

/// Common entry point of KDA and MLA blocks so one decoder layer can hold either under `self_attn`.
protocol SharpAIKimiK3TokenMixer {
    func mix(_ x: MLXArray, cache: KVCache?, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        -> MLXArray
}

/// KimiMLAAttention: multi-head latent attention with NoPE (no rotary embedding) and an optional sigmoid output gate.
/// Values are zero-padded to the query head dim and the padding is cut afterwards, as in the flash path of the
/// reference, so the score scale is `(qk_nope + qk_rope) ** -0.5`.
final class SharpAIKimiK3MLAAttention: Module, SharpAIKimiK3TokenMixer {
    let numHeads: Int
    let nopeDim: Int
    let ropeDim: Int
    let valueDim: Int
    let kvLoraRank: Int
    let scale: Float
    let useOutputGate: Bool

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayernorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayernorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: SharpAIKimiK3TextConfiguration) {
        numHeads = config.attentionHeads
        nopeDim = config.qkNopeHeadDim ?? 0
        ropeDim = config.qkRopeHeadDim ?? 0
        valueDim = config.vHeadDim ?? 0
        kvLoraRank = config.kvLoraRank ?? 0
        useOutputGate = config.mlaUseOutputGate
        let queryHeadDim = nopeDim + ropeDim
        scale = pow(Float(queryHeadDim), -0.5)
        let hidden = config.hiddenSize

        _qAProj.wrappedValue = Linear(hidden, config.qLoraRank ?? 0, bias: false)
        _qALayernorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank ?? 0, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.qLoraRank ?? 0, numHeads * queryHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(hidden, kvLoraRank + ropeDim, bias: false)
        _kvALayernorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _kvBProj.wrappedValue = Linear(kvLoraRank, numHeads * (nopeDim + valueDim), bias: false)
        _gProj.wrappedValue = config.mlaUseOutputGate
            ? Linear(hidden, numHeads * valueDim, bias: false) : nil
        _oProj.wrappedValue = Linear(numHeads * valueDim, hidden, bias: false)
        super.init()
    }

    func mix(_ x: MLXArray, cache: KVCache?, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        -> MLXArray
    {
        let B = x.dim(0)
        let S = x.dim(1)
        let queryHeadDim = nopeDim + ropeDim

        // queries: q_b_proj(q_a_layernorm(q_a_proj(x))), split into nope/rope parts that are never rotated.
        let queries = qBProj(qALayernorm(qAProj(x)))
            .reshaped(B, S, numHeads, queryHeadDim).transposed(0, 2, 1, 3)  // [B, H, S, Dq]

        // keys: kv_b_proj(kv_a_layernorm(c_kv)) gives the nope keys and the values; the rope key is shared by heads.
        let compressed = kvAProjWithMqa(x)  // [B, S, kvLora + rope]
        let kvPass = compressed[0..., 0..., 0 ..< kvLoraRank]
        let kRot = compressed[0..., 0..., kvLoraRank ..< (kvLoraRank + ropeDim)]
            .reshaped(B, 1, S, ropeDim)
        let kvExpanded = kvBProj(kvALayernorm(kvPass))
            .reshaped(B, S, numHeads, nopeDim + valueDim).transposed(0, 2, 1, 3)
        let kNope = kvExpanded[.ellipsis, 0 ..< nopeDim]
        var values = kvExpanded[.ellipsis, nopeDim ..< (nopeDim + valueDim)]  // [B, H, S, Dv]
        let kRotHeads = MLX.broadcast(kRot, to: [B, numHeads, S, ropeDim])
        let keys = concatenated([kNope, kRotHeads], axis: -1)  // [B, H, S, Dq]

        // Zero-pad values to the query head dim for the fused kernel, then cut the output back to Dv.
        let padWidth = queryHeadDim - valueDim
        if padWidth > 0 {
            values = MLX.padded(
                values,
                widths: [.init((0, 0)), .init((0, 0)), .init((0, 0)), .init((0, padWidth))])
        }
        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask)
        if padWidth > 0 {
            output = output[.ellipsis, 0 ..< valueDim]
        }

        var merged = output.transposed(0, 2, 1, 3).reshaped(B, S, numHeads * valueDim)
        if let gProj {
            merged = merged * sigmoid(gProj(x))
        }
        return oProj(merged)
    }
}

/// KimiDeltaAttention: Kimi Delta Attention with a short causal depthwise conv on q, k and v, a full-rank output
/// gate and an RMS norm gated by sigmoid. Decay is `exp(g)` with `g = lower_bound * sigmoid(exp(A_log) * (f + dt))`
/// (fla `fused_kda_gate`, lower_bound path), or `-exp(A_log) * softplus(f + dt)` when no lower bound is set.
/// Cache: `MambaCache[0]` holds the conv state of the concatenated q/k/v channels, `[1]` the recurrent state.
final class SharpAIKimiK3KDAAttention: Module, SharpAIKimiK3TokenMixer {
    let numHeads: Int
    let headDim: Int
    let projectionSize: Int
    let convKernelSize: Int
    let gateLowerBound: Float?
    let eps: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear?
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear?
    /// Depthwise conv over concatenated q, k, v channels. Sanitize merges the split checkpoint weights.
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: SharpAIKimiK3TextConfiguration) {
        let linear = config.linearAttnConfig
        numHeads = linear?.numHeads ?? config.attentionHeads
        headDim = linear?.headDim ?? (config.hiddenSize / max(1, config.attentionHeads))
        projectionSize = numHeads * headDim
        convKernelSize = linear?.shortConvKernelSize ?? 4
        gateLowerBound = linear?.gateLowerBound
        eps = config.rmsNormEps
        let hidden = config.hiddenSize

        _qProj.wrappedValue = Linear(hidden, projectionSize, bias: false)
        _kProj.wrappedValue = Linear(hidden, projectionSize, bias: false)
        _vProj.wrappedValue = Linear(hidden, projectionSize, bias: false)
        _bProj.wrappedValue = Linear(hidden, numHeads, bias: false)
        _fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionSize, bias: false)

        let fullRankGate = linear?.useFullRankGate ?? false
        _gProj.wrappedValue = fullRankGate ? Linear(hidden, projectionSize, bias: false) : nil
        _gAProj.wrappedValue = fullRankGate ? nil : Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = fullRankGate ? nil : Linear(headDim, projectionSize, bias: false)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: 3 * projectionSize,
            outputChannels: 3 * projectionSize,
            kernelSize: convKernelSize,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 3 * projectionSize,
            bias: false)
        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionSize, hidden, bias: false)

        // Placeholder initial values: the checkpoint always provides A_log and dt_bias. The reference draws
        // A_log = log(U(1, 16)) at init, which only matters when training from scratch.
        _aLog.wrappedValue = MLXArray.zeros([numHeads], dtype: .float32)
        _dtBias.wrappedValue = MLXArray.zeros([projectionSize], dtype: .float32)
        super.init()
    }

    func mix(_ x: MLXArray, cache: KVCache?, mask: MLXFast.ScaledDotProductAttentionMaskMode)
        -> MLXArray
    {
        let mamba = cache as? MambaCache
        let B = x.dim(0)
        let S = x.dim(1)
        let P = projectionSize
        let H = numHeads
        let D = headDim

        // Short causal conv over the concatenated q, k, v projections (silu after the conv).
        let qkv = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)  // [B, S, 3P]
        let convState =
            mamba?[0] ?? MLXArray.zeros([B, convKernelSize - 1, 3 * P], dtype: x.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)
        let convOut = silu(conv1d(convInput))  // [B, S, 3P]
        let newConvState = contiguous(convInput[0..., (1 - convKernelSize)..., 0...])

        let parts = MLX.split(convOut, indices: [P, 2 * P], axis: -1)
        let q = sharpAIKimiK3L2Normalize(parts[0].reshaped(B, S, H, D))
            * MLXArray(Float(pow(Float(D), -0.5))).asType(x.dtype)
        let k = sharpAIKimiK3L2Normalize(parts[1].reshaped(B, S, H, D))
        let v = parts[2].reshaped(B, S, H, D)

        // Decay: per-key log gate from the low-rank f branch, bounded or softplus-scaled.
        let rawGate = (fBProj(fAProj(x)).reshaped(B, S, H, D).asType(.float32)
            + dtBias.asType(.float32).reshaped(H, D))
        let aExp = MLX.exp(aLog.asType(.float32)).reshaped(1, 1, H, 1)
        let logDecay: MLXArray
        if let gateLowerBound {
            logDecay = MLXArray(gateLowerBound) * sigmoid(aExp * rawGate)
        } else {
            // softplus with the usual linear branch above 20, as in the reference.
            let softplusGate = MLX.where(
                MLX.greater(rawGate, MLXArray(Float(20))), rawGate,
                MLX.log(MLX.exp(rawGate) + MLXArray(Float(1))))
            logDecay = MLXArray(Float(-1)) * aExp * softplusGate
        }
        let decay = MLX.exp(logDecay)  // [B, S, H, D] multiplier
        let beta = sigmoid(bProj(x).asType(.float32))  // [B, S, H]

        let (y, newRecState) = SharpAIOverlayGatedDeltaKDA.update(
            q: q, k: k, v: v, g: decay, beta: beta,
            state: mamba?[1], mask: nil, useKernel: !training)

        if let mamba {
            mamba[0] = newConvState
            mamba[1] = newRecState
            mamba.advance(S)
        }

        let gate: MLXArray
        if let gProj {
            gate = gProj(x).reshaped(B, S, H, D)
        } else {
            gate = gBProj!(gAProj!(x)).reshaped(B, S, H, D)
        }
        let normed = oNorm(y.asType(x.dtype)) * sigmoid(gate)
        return oProj(normed.reshaped(B, S, -1))
    }
}

/// Row-wise L2 norm with eps inside the square root, as in fla `l2norm_fwd` (eps 1e-6).
func sharpAIKimiK3L2Normalize(_ x: MLXArray) -> MLXArray {
    let sumSquares = MLX.sum(MLX.square(x.asType(.float32)), axis: -1, keepDims: true)
    return (x.asType(.float32) * MLX.rsqrt(sumSquares + MLXArray(Float(1e-6)))).asType(x.dtype)
}

// MARK: - Feed-forward

/// KimiMLP (dense layer 0 and the shared experts): down(act(gate(x), up(x))).
final class SharpAIKimiK3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let activation: SharpAIKimiK3Activation

    init(hidden: Int, intermediate: Int, activation: SharpAIKimiK3Activation) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gate: gateProj(x), up: upProj(x)))
    }
}

/// KimiMoEGate: sigmoid (or softmax) scores; experts are chosen on scores plus e_score_correction_bias; the
/// weights are the unbiased scores of the chosen experts, renormalized when moe_renormalize, then scaled.
final class SharpAIKimiK3Router: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray  // [E, hidden]
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray  // [E]
    let topK: Int
    let renormalize: Bool
    let routedScalingFactor: Float
    let usesSoftmax: Bool

    init(_ config: SharpAIKimiK3TextConfiguration, numExperts: Int, topK: Int) {
        self.topK = topK
        renormalize = config.moeRenormalize
        routedScalingFactor = config.routedScalingFactor
        usesSoftmax = config.moeRouterActivationFunc == "softmax"
        _weight.wrappedValue = MLXArray.zeros([numExperts, config.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([numExperts], dtype: .float32)
        super.init()
    }

    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        let logits = MLX.matmul(x.asType(.float32), weight.asType(.float32).T)  // [B, S, E]
        let scores = usesSoftmax ? MLX.softmax(logits, axis: -1) : sigmoid(logits)
        let selection = scores + correctionBias.asType(.float32)
        let indices = MLX.argPartition(
            selection * MLXArray(Float(-1)), kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
        var weights = MLX.takeAlong(scores, indices, axis: -1)
        if topK > 1 && renormalize {
            weights = weights / (MLX.sum(weights, axis: -1, keepDims: true) + MLXArray(Float(1e-20)))
        }
        return (indices.asType(.int32), weights * MLXArray(routedScalingFactor))
    }
}

/// KimiSparseMoeBlock. Routed experts run at `routed_expert_hidden_size` when set (latent MoE): the hidden state is
/// down-projected, mixed by the experts, optionally normalized, then up-projected. Shared experts read the input
/// of the block, not the latent state.
final class SharpAIKimiK3SparseMoE: Module {
    let numExperts: Int
    let hasLatent: Bool

    @ModuleInfo(key: "gate") var gate: SharpAIKimiK3Router
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIKimiK3MLP?
    @ModuleInfo(key: "routed_expert_down_proj") var routedExpertDownProj: Linear?
    @ModuleInfo(key: "routed_expert_up_proj") var routedExpertUpProj: Linear?
    @ModuleInfo(key: "routed_expert_norm") var routedExpertNorm: RMSNorm?

    init(_ config: SharpAIKimiK3TextConfiguration) {
        let hidden = config.hiddenSize
        let count = config.numExperts ?? 0
        let topK = config.numExpertsPerToken ?? 1
        let intermediate = config.moeIntermediateSize ?? 0
        let activation = SharpAIKimiK3Activation(config)
        numExperts = count
        hasLatent = config.routedExpertHiddenSize != nil
        let latentWidth = config.routedExpertHiddenSize ?? hidden

        _gate.wrappedValue = SharpAIKimiK3Router(config, numExperts: count, topK: topK)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: latentWidth, hiddenDims: intermediate, numExperts: count,
            activation: MLXNN.silu,
            activationProduct: { gate, up in activation(gate: gate, up: up) })
        _sharedExperts.wrappedValue =
            config.numSharedExperts > 0
            ? SharpAIKimiK3MLP(
                hidden: hidden, intermediate: intermediate * config.numSharedExperts,
                activation: activation)
            : nil
        _routedExpertDownProj.wrappedValue =
            hasLatent ? Linear(hidden, latentWidth, bias: false) : nil
        _routedExpertUpProj.wrappedValue =
            hasLatent ? Linear(latentWidth, hidden, bias: false) : nil
        _routedExpertNorm.wrappedValue =
            hasLatent && config.latentMoeUseNorm
            ? RMSNorm(dimensions: latentWidth, eps: config.rmsNormEps) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate.route(x)  // [B, S, K]
        var latent = x
        if let routedExpertDownProj {
            latent = routedExpertDownProj(latent)
        }

        // Gather dispatch: only the top-k experts of each token run; combine with the routing weights.
        let expertOut = switchMLP(latent, indices)  // [B, S, K, latent]
        var y = MLX.sum(
            expertOut * weights.asType(expertOut.dtype).expandedDimensions(axis: -1), axis: -2)

        if let routedExpertNorm {
            y = routedExpertNorm(y)
        }
        if let routedExpertUpProj {
            y = routedExpertUpProj(y)
        }
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder layer

/// KimiDecoderLayer. With `attn_res_block_size` set, each layer reads the completed-block bank plus the open
/// partial sum (`_forward_attn_residual`); otherwise it is a plain pre-norm residual block.
final class SharpAIKimiK3DecoderLayer: Module {
    let layerIndex: Int
    let attnResBlockSize: Int?
    let eps: Float

    @ModuleInfo(key: "self_attn") var selfAttn: Module & SharpAIKimiK3TokenMixer
    @ModuleInfo(key: "mlp") var mlp: SharpAIKimiK3MLP?
    @ModuleInfo(key: "block_sparse_moe") var sparseMoe: SharpAIKimiK3SparseMoE?
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "self_attention_res_norm") var selfAttentionResNorm: SharpAIKimiK3NormWeight?
    @ModuleInfo(key: "self_attention_res_proj") var selfAttentionResProj: Linear?
    @ModuleInfo(key: "mlp_res_norm") var mlpResNorm: SharpAIKimiK3NormWeight?
    @ModuleInfo(key: "mlp_res_proj") var mlpResProj: Linear?

    init(_ config: SharpAIKimiK3TextConfiguration, layerIndex: Int) {
        self.layerIndex = layerIndex
        attnResBlockSize = config.attnResBlockSize
        eps = config.rmsNormEps
        let hidden = config.hiddenSize
        let activation = SharpAIKimiK3Activation(config)

        if config.isKDALayer(layerIndex) {
            _selfAttn.wrappedValue = SharpAIKimiK3KDAAttention(config)
        } else {
            _selfAttn.wrappedValue = SharpAIKimiK3MLAAttention(config)
        }

        let isMoE =
            config.numExperts != nil && layerIndex >= config.firstKDenseReplace
            && layerIndex % max(1, config.moeLayerFreq) == 0
        _sparseMoe.wrappedValue = isMoE ? SharpAIKimiK3SparseMoE(config) : nil
        _mlp.wrappedValue =
            isMoE ? nil
            : SharpAIKimiK3MLP(hidden: hidden, intermediate: config.intermediateSize, activation: activation)

        _inputLayernorm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)
        _postAttentionLayernorm.wrappedValue = RMSNorm(dimensions: hidden, eps: config.rmsNormEps)

        if config.attnResBlockSize != nil {
            _selfAttentionResNorm.wrappedValue = SharpAIKimiK3NormWeight(hidden)
            _selfAttentionResProj.wrappedValue = Linear(hidden, 1, bias: false)
            _mlpResNorm.wrappedValue = SharpAIKimiK3NormWeight(hidden)
            _mlpResProj.wrappedValue = Linear(hidden, 1, bias: false)
        }
        super.init()
    }

    private func feedForward(_ x: MLXArray) -> MLXArray {
        if let sparseMoe {
            return sparseMoe(x)
        }
        return mlp!(x)
    }

    /// `blocks` is the completed-block bank shared by the layers of one forward pass; it is updated in place.
    func forward(
        _ stream: MLXArray, blocks: inout [MLXArray], cache: KVCache?,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let attnResBlockSize, let selfAttentionResNorm, let selfAttentionResProj,
            let mlpResNorm, let mlpResProj
        else {
            let attnOut = selfAttn.mix(inputLayernorm(stream), cache: cache, mask: mask)
            let afterAttn = stream + attnOut
            return afterAttn + feedForward(postAttentionLayernorm(afterAttn))
        }

        // Attention input: mix the bank and the open partial sum (the layer input) when the bank is not empty.
        var attnInput = stream
        if !blocks.isEmpty {
            attnInput = sharpAIKimiK3AttnRes(
                sources: blocks + [stream], proj: selfAttentionResProj.weight,
                norm: selfAttentionResNorm, eps: eps)
        }

        // A block closes before layers whose index is a multiple of the block size. The layer input is
        // appended as a finished block and the new partial sum starts from this layer's attention output.
        var partialBeforeAttn: MLXArray? = stream
        if layerIndex % attnResBlockSize == 0 {
            blocks.append(stream)
            partialBeforeAttn = nil
        }

        let attnOut = selfAttn.mix(inputLayernorm(attnInput), cache: cache, mask: mask)
        let afterAttn = partialBeforeAttn.map { $0 + attnOut } ?? attnOut

        let mlpInput = sharpAIKimiK3AttnRes(
            sources: blocks + [afterAttn], proj: mlpResProj.weight, norm: mlpResNorm, eps: eps)
        let mlpOut = feedForward(postAttentionLayernorm(mlpInput))
        return afterAttn + mlpOut
    }
}

// MARK: - Backbone

/// KimiLinearModel: embeddings, decoder layers, the output attention-residual mix and the final RMS norm.
final class SharpAIKimiK3Backbone: Module {
    let attnResBlockSize: Int?
    let eps: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIKimiK3DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "output_attn_res_norm") var outputAttnResNorm: SharpAIKimiK3NormWeight?
    @ModuleInfo(key: "output_attn_res_proj") var outputAttnResProj: Linear?

    init(_ config: SharpAIKimiK3TextConfiguration) {
        attnResBlockSize = config.attnResBlockSize
        eps = config.rmsNormEps
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            SharpAIKimiK3DecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.attnResBlockSize != nil {
            _outputAttnResNorm.wrappedValue = SharpAIKimiK3NormWeight(config.hiddenSize)
            _outputAttnResProj.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        var stream = embedTokens(tokens)
        let caches: [KVCache?]
        if let cache, cache.count == layers.count {
            caches = cache.map { Optional($0) }
        } else {
            caches = Array(repeating: nil, count: layers.count)
        }

        var blocks: [MLXArray] = []
        for (index, layer) in layers.enumerated() {
            let mask = createAttentionMask(h: stream, cache: caches[index])
            stream = layer.forward(stream, blocks: &blocks, cache: caches[index], mask: mask)
        }

        if let outputAttnResNorm, let outputAttnResProj {
            stream = sharpAIKimiK3AttnRes(
                sources: blocks + [stream], proj: outputAttnResProj.weight,
                norm: outputAttnResNorm, eps: eps)
        }
        return norm(stream)
    }
}

/// The `language_model` node of the checkpoint: backbone plus LM head (not tied).
final class SharpAIKimiK3TextModel: Module {
    @ModuleInfo(key: "model") var model: SharpAIKimiK3Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: SharpAIKimiK3TextConfiguration) {
        _model.wrappedValue = SharpAIKimiK3Backbone(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(tokens, cache: cache))
    }
}

// MARK: - Model

/// Text-only kimi_k3 model. Checkpoint keys are `language_model.model.*` and `language_model.lm_head.*`.
/// The vision tower and mm_projector are not ported; sanitize drops their keys.
public class SharpAIKimiK3Model: Module, LLMModel, LoRAModel {
    public let vocabularySize: Int
    let configuration: SharpAIKimiK3TextConfiguration

    @ModuleInfo(key: "language_model") var languageModel: SharpAIKimiK3TextModel

    public init(_ configuration: SharpAIKimiK3TextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        _languageModel.wrappedValue = SharpAIKimiK3TextModel(configuration)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// KDA layers get a MambaCache (conv and recurrent state); full-attention (MLA) layers get KV caches.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try (0 ..< configuration.hiddenLayers).map { index -> KVCache in
            if configuration.isKDALayer(index) {
                return MambaCache()
            }
            return try makeAttentionKVCache(parameters: parameters)
        }
    }

    public func makeCache() -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { index -> KVCache in
            configuration.isKDALayer(index) ? MambaCache() : KVCacheSimple()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIKimiK3Sanitizer.sanitize(weights)
    }
}

extension SharpAIKimiK3Model {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
