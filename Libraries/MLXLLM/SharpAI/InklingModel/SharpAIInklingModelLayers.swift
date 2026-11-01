// SharpAI fork overlay: inkling_model text building blocks. Not part of upstream mlx-swift-lm.
//
// Ported from transformers `models/inkling/modeling_inkling.py` (text path only). Modules:
// - RMSNorm with a plain scale (no zero-centering).
// - Depthwise causal short convolution with residual (the four "sconv" streams: keys, values,
//   attention output, MLP output).
// - Hidden-state conditioned relative position bias, zero beyond `rel_extent`.
// - Sink MoE: sigmoid router, routed + shared experts, shared logits normalized with the routed.
// Vision, audio and MTP modules are not built.
//
// Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/inkling/modeling_inkling.py
// (RMSNorm, relative logits, attention with log scaling and the sliding mask, short convolution,
// router, shared experts and decoder layer were compared line by line with the Swift code).
// Still unverified: no forward pass has been compared numerically with the reference, because no
// build was run for this change. The test suite checks the formulas on tiny inputs.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMSNorm with a plain scale: `x * rsqrt(mean(x^2) + eps) * weight`.
final class SharpAIInklingRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Depthwise causal convolution over time, with residual:
/// `y[t, c] = x[t, c] + sum_j w[c, 0, j] * x[t - (K - 1) + j, c]`, zero history before token 0.
/// Matches transformers `InklingShortConvolution` (fp32 path, no cache, no padding mask).
final class SharpAIInklingShortConv: Module {
    /// `[channels, 1, kernel]`, the same layout as the depthwise `nn.Conv1d` weight.
    @ParameterInfo(key: "weight") var weight: MLXArray
    let kernelSize: Int

    init(channels: Int, kernelSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([channels, 1, kernelSize])
        self.kernelSize = kernelSize
    }

    /// `x` is `[batch, length, channels]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let channels = x.dim(2)
        let input = x.asType(.float32)
        let history = MLXArray.zeros([batch, kernelSize - 1, channels], dtype: .float32)
        let padded = MLX.concatenated([history, input], axis: 1)
        let w = weight.asType(.float32)

        var output = input
        for j in 0 ..< kernelSize {
            let window = padded[0..., j ..< (j + length), 0...]
            output = output + window * w[0..., 0, j]
        }
        return output.asType(x.dtype)
    }
}

/// Relative position bias conditioned on the query's relative states (transformers
/// `InklingRelativeLogits`). `proj` is a bank of bias-versus-distance profiles. A query's
/// relative states mix them into one bias per backward distance. The bias is zero for
/// `distance < 0` and for `distance >= relExtent`. Causality stays in the attention mask.
final class SharpAIInklingRelativeLogits: Module {
    /// `[dRel, relExtent]`.
    @ParameterInfo(key: "proj") var proj: MLXArray
    let relExtent: Int

    init(dRel: Int, relExtent: Int) {
        _proj.wrappedValue = MLXArray.zeros([dRel, relExtent])
        self.relExtent = relExtent
    }

    /// - Parameters:
    ///   - relativeStates: `[batch, queryLength, heads, dRel]`.
    ///   - queryPositions: `[queryLength]`, absolute positions of the queries.
    ///   - keyPositions: `[keyLength]`, absolute positions of the keys.
    /// - Returns: `[batch, heads, queryLength, keyLength]`.
    func positionBias(
        relativeStates: MLXArray, queryPositions: MLXArray, keyPositions: MLXArray
    ) -> MLXArray {
        // [B, T, H, E] -> [B, H, T, E]
        let relLogits = matmul(relativeStates, proj).transposed(0, 2, 1, 3)
        let distance =
            queryPositions.expandedDimensions(axis: 1)
            - keyPositions.expandedDimensions(axis: 0)

        let zero = MLXArray(Int32(0))
        let lastExtent = MLXArray(Int32(relExtent - 1))
        let gatherIndex = MLX.minimum(MLX.maximum(distance, zero), lastExtent)
        let index = broadcast(
            gatherIndex,
            to: [relLogits.dim(0), relLogits.dim(1), distance.dim(0), distance.dim(1)])
        let gathered = takeAlong(relLogits, index, axis: -1)

        // In range means `0 <= distance < relExtent`.
        let beyondExtent = greaterEqual(distance, MLXArray(Int32(relExtent)))
        let belowExtent = MLX.where(beyondExtent, MLXArray(false), MLXArray(true))
        let inRange = MLX.where(greaterEqual(distance, zero), belowExtent, MLXArray(false))
        let masked = MLX.where(inRange, gathered, MLXArray(Float(0)))
        return masked.asType(relLogits.dtype)
    }
}

/// Self-attention with the relative position bias, QK-norm, and the k/v short convolutions.
/// Sliding layers attend over `slidingWindow` keys and use the window as bias extent. Full layers
/// use `relExtent`, and they log-scale queries and bias past `logScalingNFloor`.
final class SharpAIInklingAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let dRel: Int
    let slidingWindow: Int?
    let logScalingNFloor: Int?
    let logScalingAlpha: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIInklingRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIInklingRMSNorm
    @ModuleInfo(key: "k_sconv") var kSconv: SharpAIInklingShortConv
    @ModuleInfo(key: "v_sconv") var vSconv: SharpAIInklingShortConv
    @ModuleInfo(key: "rel_logits_proj") var relLogitsProj: SharpAIInklingRelativeLogits

    init(_ config: SharpAIInklingTextConfiguration, layer: Int) {
        let sliding = config.slidingLayers[layer]
        numHeads = sliding ? config.swaAttentionHeads : config.attentionHeads
        numKVHeads = sliding ? config.swaKVHeads : config.kvHeads
        headDim = sliding ? config.swaHeadDim : config.headDim
        dRel = config.dRel
        slidingWindow = sliding ? config.slidingWindow : nil
        logScalingNFloor = sliding ? nil : config.logScalingNFloor
        logScalingAlpha = config.logScalingAlpha

        let queryWidth = numHeads * headDim
        let kvWidth = numKVHeads * headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, queryWidth, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, kvWidth, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, kvWidth, bias: false)
        _rProj.wrappedValue = Linear(config.hiddenSize, numHeads * config.dRel, bias: false)
        _oProj.wrappedValue = Linear(queryWidth, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = SharpAIInklingRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIInklingRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kSconv.wrappedValue = SharpAIInklingShortConv(
            channels: kvWidth, kernelSize: config.sconvKernelSize)
        _vSconv.wrappedValue = SharpAIInklingShortConv(
            channels: kvWidth, kernelSize: config.sconvKernelSize)
        _relLogitsProj.wrappedValue = SharpAIInklingRelativeLogits(
            dRel: config.dRel, relExtent: sliding ? config.slidingWindow : config.relExtent)
    }

    /// `x` is `[batch, length, hidden]`. Full-sequence forward, no cache.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var q = qNorm(qProj(x).reshaped([batch, length, numHeads, headDim]))
        var k = kNorm(kSconv(kProj(x)).reshaped([batch, length, numKVHeads, headDim]))
        let v = vSconv(vProj(x)).reshaped([batch, length, numKVHeads, headDim])
        let relativeStates = rProj(x).reshaped([batch, length, numHeads, dRel])

        let positions = MLXArray((0 ..< length).map { Int32($0) })
        var bias = relLogitsProj.positionBias(
            relativeStates: relativeStates, queryPositions: positions, keyPositions: positions)

        // Log scaling of full-attention logits past the floor, as transformers does in f32.
        if let floor = logScalingNFloor {
            let effective = positions.asType(.float32) + Float(1)
            let ratio = MLX.maximum(effective / Float(floor), MLXArray(Float(1)))
            let tau = (MLXArray(Float(1)) + Float(logScalingAlpha) * MLX.log(ratio))
                .reshaped([1, 1, length, 1])
            q = (q.asType(.float32) * tau.transposed(0, 2, 1, 3)).asType(q.dtype)
            bias = (bias.asType(.float32) * tau).asType(bias.dtype)
        }

        q = q.transposed(0, 2, 1, 3).asType(.float32)
        k = k.transposed(0, 2, 1, 3).asType(.float32)
        let vHeads = v.transposed(0, 2, 1, 3).asType(.float32)

        let groups = numHeads / numKVHeads
        if groups > 1 {
            k = MLX.repeated(k, count: groups, axis: 1)
        }
        let valueHeads = groups > 1 ? MLX.repeated(vHeads, count: groups, axis: 1) : vHeads

        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * (1.0 / Float(headDim))
        scores = scores + bias.asType(.float32) + causalMask(length: length)
        let probs = softmax(scores, axis: -1)
        let output = matmul(probs, valueHeads)

        let merged = output.transposed(0, 2, 1, 3).reshaped([batch, length, numHeads * headDim])
        return oProj(merged.asType(x.dtype))
    }

    /// Additive mask: 0 where the key is visible, -inf elsewhere. Visible means causal, and
    /// within `slidingWindow` for sliding layers (`kv > q - window`, as transformers does).
    private func causalMask(length: Int) -> MLXArray {
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let distance =
            positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
        let zero = MLXArray(Int32(0))
        let negInf = MLXArray(-Float.infinity)
        var visible = MLX.where(
            greaterEqual(distance, zero), MLXArray(Float(0)), negInf)
        if let window = slidingWindow {
            // Outside the window means `distance >= window`, i.e. `kv <= q - window`.
            let outside = MLX.where(
                greaterEqual(distance, MLXArray(Int32(window))), negInf, MLXArray(Float(0)))
            visible = visible + outside
        }
        return visible
    }
}

/// Dense SwiGLU MLP with a scalar `global_scale` (transformers `InklingMLP`).
final class SharpAIInklingDenseMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _globalScale.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x)) * globalScale
    }
}

/// Router of the sink MoE (transformers `InklingTopkRouter`). Scores are `sigmoid(logits)`. The
/// top-k choice adds `e_score_correction_bias`. The top-k routed logits and the shared logits are
/// normalized together, so the shared experts sink probability mass in the expert weights.
final class SharpAIInklingRouter: Module {
    /// `[routed + shared, hidden]`.
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray
    let routedCount: Int
    let sharedCount: Int
    let topK: Int
    let routeScale: Float

    init(_ config: SharpAIInklingTextConfiguration) {
        routedCount = config.routedExperts
        sharedCount = config.sharedExperts
        topK = config.expertsPerToken
        routeScale = config.routeScale
        _weight.wrappedValue = MLXArray.zeros([routedCount + sharedCount, config.hiddenSize])
        _globalScale.wrappedValue = MLXArray.ones([1])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([routedCount])
    }

    /// `x` is `[batch, length, hidden]`. Returns routed indices `[B, T, K]`, routed weights
    /// `[B, T, K]`, and shared gammas `[B, T, S]`.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray, gammas: MLXArray) {
        let logits = matmul(x, weight.T)
        let routedLogits = logits[.ellipsis, ..<routedCount]
        let sharedLogits = logits[.ellipsis, routedCount...]

        let choice = sigmoid(routedLogits) + eScoreCorrectionBias
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        let chosen = takeAlong(routedLogits, indices, axis: -1)
        let combined = MLX.concatenated([chosen, sharedLogits], axis: -1)
        // exp(logsigmoid(z) - logsumexp(logsigmoid(z))) is sigmoid(z) / sum(sigmoid(z)).
        let activated = sigmoid(combined)
        let normalized = activated / activated.sum(axis: -1, keepDims: true)
        let scaled = normalized * (globalScale * routeScale)

        return (
            indices: indices,
            weights: scaled[.ellipsis, ..<topK],
            gammas: scaled[.ellipsis, topK...]
        )
    }
}

/// Sparse MoE block: routed experts plus shared experts, all as `SwitchGLU` banks.
final class SharpAIInklingMoE: Module {
    @ModuleInfo(key: "gate") var gate: SharpAIInklingRouter
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwitchGLU
    let sharedCount: Int

    init(_ config: SharpAIInklingTextConfiguration) {
        sharedCount = config.sharedExperts
        _gate.wrappedValue = SharpAIInklingRouter(config)
        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.routedExperts)
        _sharedExperts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.sharedExperts)
    }

    /// `x` is `[batch, length, hidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights, gammas) = gate(x)
        let routed = (experts(x, indices) * weights[.ellipsis, .newAxis]).sum(axis: -2)

        // Every shared expert runs on every token. Its gamma scales its output.
        let sharedIndices = broadcast(
            MLXArray((0 ..< sharedCount).map { UInt32($0) }),
            to: [x.dim(0), x.dim(1), sharedCount])
        let shared = (sharedExperts(x, sharedIndices) * gammas[.ellipsis, .newAxis]).sum(axis: -2)
        return routed + shared
    }
}

/// One decoder layer: attention, then a dense MLP or a sparse MoE. Each sub-block output passes
/// through a short convolution before its residual add, as in transformers.
final class SharpAIInklingDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIInklingAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIInklingRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SharpAIInklingRMSNorm
    @ModuleInfo(key: "attn_sconv") var attnSconv: SharpAIInklingShortConv
    @ModuleInfo(key: "mlp_sconv") var mlpSconv: SharpAIInklingShortConv
    @ModuleInfo(key: "mlp") var denseMLP: SharpAIInklingDenseMLP?
    @ModuleInfo(key: "moe") var moe: SharpAIInklingMoE?

    init(_ config: SharpAIInklingTextConfiguration, layer: Int) {
        _selfAttn.wrappedValue = SharpAIInklingAttention(config, layer: layer)
        _inputLayerNorm.wrappedValue = SharpAIInklingRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIInklingRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnSconv.wrappedValue = SharpAIInklingShortConv(
            channels: config.hiddenSize, kernelSize: config.sconvKernelSize)
        _mlpSconv.wrappedValue = SharpAIInklingShortConv(
            channels: config.hiddenSize, kernelSize: config.sconvKernelSize)
        if config.denseLayers[layer] {
            _denseMLP.wrappedValue = SharpAIInklingDenseMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
            _moe.wrappedValue = nil
        } else {
            _denseMLP.wrappedValue = nil
            _moe.wrappedValue = SharpAIInklingMoE(config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attnSconv(selfAttn(inputLayerNorm(x)))
        let normed = postAttentionLayerNorm(h)
        let feedForward: MLXArray
        if let moe {
            feedForward = moe(normed)
        } else if let denseMLP {
            feedForward = denseMLP(normed)
        } else {
            preconditionFailure("inkling layer has neither a dense MLP nor a MoE block")
        }
        return h + mlpSconv(feedForward)
    }
}
