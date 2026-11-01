// SharpAI fork overlay: inkling_mm_model text backbone. Not part of upstream mlx-swift-lm.
//
// Ported from transformers `models/inkling/modeling_inkling.py` (Verified against:
// https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/inkling/modeling_inkling.py).
// Module keys are the checkpoint names (`attn.wq_du`, `mlp.w13_dn`, `mlp.experts.w13_weight`, ...),
// so the weights load with no renaming. Full-sequence forward only, with no KV cache and no short-conv state.
//
// The fused gate/up tensors interleave rows (even gate, odd up). Their output is reshaped to
// `[..., width, 2]` after the matmul, so no weight copy is needed.
// Shape facts, from the checkpoint safetensors headers: `w13_dn` is [2 * dense, hidden],
// `experts.w13_weight` is [experts, 2 * moe, hidden], and `shared_w13_weight` is [shared, 2 * moe, hidden].

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// RMSNorm with a plain scale: `x * rsqrt(mean(x^2) + eps) * weight`.
final class SharpAIInklingMmRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// Splits a fused gate/up output whose last axis is `2 * width` and interleaves gate, up, gate, up.
/// Row `2a` is gate unit `a` and row `2a + 1` is up unit `a` (transformers `Interleave` + `Chunk`).
func sharpAIInklingMmSplitGateUp(_ fused: MLXArray, width: Int) -> (gate: MLXArray, up: MLXArray) {
    var pairShape = Array(fused.shape.dropLast())
    pairShape.append(width)
    pairShape.append(2)
    let pairs = fused.reshaped(pairShape)
    return (gate: pairs[.ellipsis, 0], up: pairs[.ellipsis, 1])
}

/// Depthwise causal convolution over time, with residual:
/// `y[t, c] = x[t, c] + sum_j w[c, j] * x[t - (K - 1) + j, c]`, zero history before token 0.
/// Matches transformers `InklingShortConvolution` (fp32 path, no cache, no padding mask).
final class SharpAIInklingMmShortConv: Module {
    /// `[channels, kernel, 1]`: the MLX depthwise `Conv1d` layout. The sanitizer remaps the checkpoint layout.
    @ParameterInfo(key: "weight") var weight: MLXArray
    let kernelSize: Int

    init(channels: Int, kernelSize: Int) {
        _weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
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
            output = output + window * w[0..., j, 0]
        }
        return output.asType(x.dtype)
    }
}

/// Relative position bias conditioned on the query's relative states (transformers
/// `InklingRelativeLogits`). `proj` is a bank of bias-versus-distance profiles. Each query's relative
/// states mix them into one bias per backward distance. The bias is zero for `distance < 0` and for
/// `distance >= relExtent`. Causality stays in the attention mask.
final class SharpAIInklingMmRelativeLogits: Module {
    /// `[dRel, relExtent]`, the checkpoint layout.
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

/// Self-attention with the relative position bias, QK-norm and the k/v short convolutions.
/// Sliding layers attend over the last `slidingWindowSize` keys, and their bias extent is that window.
/// Full layers use `relExtent`, and they log-scale queries and bias past `logScalingNFloor`.
final class SharpAIInklingMmAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let dRel: Int
    let slidingWindow: Int?
    let logScalingNFloor: Int?
    let logScalingAlpha: Float

    @ModuleInfo(key: "wq_du") var wqDu: Linear
    @ModuleInfo(key: "wk_dv") var wkDv: Linear
    @ModuleInfo(key: "wv_dv") var wvDv: Linear
    @ModuleInfo(key: "wr_du") var wrDu: Linear
    @ModuleInfo(key: "wo_ud") var woUd: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "k_sconv") var kSconv: SharpAIInklingMmShortConv
    @ModuleInfo(key: "v_sconv") var vSconv: SharpAIInklingMmShortConv
    @ModuleInfo(key: "rel_logits_proj") var relLogitsProj: SharpAIInklingMmRelativeLogits

    init(_ config: SharpAIInklingMmModelTextConfiguration, layer: Int) {
        let sliding = config.layerTypes[layer] == .hybridSliding
        numHeads = sliding ? config.swaAttentionHeads : config.attentionHeads
        numKVHeads = sliding ? config.swaKVHeads : config.kvHeads
        headDim = sliding ? config.swaHeadDim : config.headDim
        dRel = config.dRel
        slidingWindow = sliding ? config.slidingWindowSize : nil
        logScalingNFloor = sliding ? nil : config.logScalingNFloor
        logScalingAlpha = config.logScalingAlpha

        let queryWidth = numHeads * headDim
        let kvWidth = numKVHeads * headDim
        _wqDu.wrappedValue = Linear(config.hiddenSize, queryWidth, bias: false)
        _wkDv.wrappedValue = Linear(config.hiddenSize, kvWidth, bias: false)
        _wvDv.wrappedValue = Linear(config.hiddenSize, kvWidth, bias: false)
        _wrDu.wrappedValue = Linear(config.hiddenSize, numHeads * config.dRel, bias: false)
        _woUd.wrappedValue = Linear(queryWidth, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = SharpAIInklingMmRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIInklingMmRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kSconv.wrappedValue = SharpAIInklingMmShortConv(
            channels: kvWidth, kernelSize: config.convKernelSize)
        _vSconv.wrappedValue = SharpAIInklingMmShortConv(
            channels: kvWidth, kernelSize: config.convKernelSize)
        _relLogitsProj.wrappedValue = SharpAIInklingMmRelativeLogits(
            dRel: config.dRel, relExtent: sliding ? config.slidingWindowSize : config.relExtent)
    }

    /// `x` is `[batch, length, hidden]`. Full-sequence forward, no cache.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        var q = qNorm(wqDu(x).reshaped([batch, length, numHeads, headDim]))
        let k = kNorm(kSconv(wkDv(x)).reshaped([batch, length, numKVHeads, headDim]))
        let v = vSconv(wvDv(x)).reshaped([batch, length, numKVHeads, headDim])
        let relativeStates = wrDu(x).reshaped([batch, length, numHeads, dRel])

        let positions = MLXArray((0 ..< length).map { Int32($0) })
        var bias = relLogitsProj.positionBias(
            relativeStates: relativeStates, queryPositions: positions, keyPositions: positions)

        // Log scaling of full-attention logits past the floor, computed in f32 as transformers does.
        if let floor = logScalingNFloor {
            let effective = positions.asType(.float32) + Float(1)
            let ratio = MLX.maximum(effective / Float(floor), MLXArray(Float(1)))
            let tau = (MLXArray(Float(1)) + Float(logScalingAlpha) * MLX.log(ratio))
                .reshaped([1, 1, length, 1])
            q = (q.asType(.float32) * tau.transposed(0, 2, 1, 3)).asType(q.dtype)
            bias = (bias.asType(.float32) * tau).asType(bias.dtype)
        }

        q = q.transposed(0, 2, 1, 3).asType(.float32)
        var kHeads = k.transposed(0, 2, 1, 3).asType(.float32)
        var vHeads = v.transposed(0, 2, 1, 3).asType(.float32)

        let groups = numHeads / numKVHeads
        if groups > 1 {
            kHeads = MLX.repeated(kHeads, count: groups, axis: 1)
            vHeads = MLX.repeated(vHeads, count: groups, axis: 1)
        }

        // q and k are RMS-normalized per head, so the scale is 1/d and not 1/sqrt(d).
        var scores = matmul(q, kHeads.transposed(0, 1, 3, 2)) * (1.0 / Float(headDim))
        scores = scores + bias.asType(.float32) + causalMask(length: length)
        let probs = softmax(scores, axis: -1)
        let output = matmul(probs, vHeads)

        let merged = output.transposed(0, 2, 1, 3).reshaped([batch, length, numHeads * headDim])
        return woUd(merged.asType(x.dtype))
    }

    /// Additive mask: 0 where the key is visible and -inf elsewhere. Visible means causal, and within
    /// `slidingWindow` for sliding layers (distance < window).
    private func causalMask(length: Int) -> MLXArray {
        let positions = MLXArray((0 ..< length).map { Int32($0) })
        let distance =
            positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)
        let zero = MLXArray(Int32(0))
        let negInf = MLXArray(-Float.infinity)
        var visible = MLX.where(greaterEqual(distance, zero), MLXArray(Float(0)), negInf)
        if let window = slidingWindow {
            let outside = MLX.where(
                greaterEqual(distance, MLXArray(Int32(window))), negInf, MLXArray(Float(0)))
            visible = visible + outside
        }
        return visible
    }
}

/// Dense SwiGLU MLP with a scalar `global_scale` (transformers `InklingMLP`).
final class SharpAIInklingMmDenseMLP: Module {
    @ModuleInfo(key: "w13_dn") var w13Dn: Linear
    @ModuleInfo(key: "w2_md") var w2Md: Linear
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray
    let width: Int

    init(hiddenSize: Int, intermediateSize: Int) {
        _w13Dn.wrappedValue = Linear(hiddenSize, 2 * intermediateSize, bias: false)
        _w2Md.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        _globalScale.wrappedValue = MLXArray.ones([1])
        width = intermediateSize
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (gate, up) = sharpAIInklingMmSplitGateUp(w13Dn(x), width: width)
        return w2Md(silu(gate) * up) * globalScale
    }
}

/// Router of the sink MoE (transformers `InklingTopkRouter`). Scores are `sigmoid(logits)`. The top-k
/// choice adds `e_score_correction_bias` (the checkpoint key `mlp.gate.bias`). The chosen routed
/// logits and the shared logits are normalized together, so the shared experts take a share of the mass.
final class SharpAIInklingMmRouter: Module {
    /// `[routed + shared, hidden]`.
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let routedCount: Int
    let sharedCount: Int
    let topK: Int
    let routeScale: Float

    init(_ config: SharpAIInklingMmModelTextConfiguration) {
        routedCount = config.routedExperts
        sharedCount = config.sharedExperts
        topK = config.numExpertsPerToken
        routeScale = config.routeScale
        _weight.wrappedValue = MLXArray.zeros([routedCount + sharedCount, config.hiddenSize])
        _globalScale.wrappedValue = MLXArray.ones([1])
        _bias.wrappedValue = MLXArray.zeros([routedCount])
    }

    /// `x` is `[batch, length, hidden]`. Returns routed indices `[B, T, K]`, routed weights `[B, T, K]`,
    /// and shared gammas `[B, T, S]`.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray, gammas: MLXArray) {
        let logits = matmul(x, weight.T)
        let routedLogits = logits[.ellipsis, ..<routedCount]
        let sharedLogits = logits[.ellipsis, routedCount...]

        let choice = sigmoid(routedLogits) + bias
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        let chosen = takeAlong(routedLogits, indices, axis: -1)
        let combined = MLX.concatenated([chosen, sharedLogits], axis: -1)
        // exp(logsigmoid(z) - logsumexp(logsigmoid(z))) equals sigmoid(z) / sum(sigmoid(z)).
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

/// Runs the gated expert MLPs for `indices` with `gatherMM`, so only the selected expert weights are read.
/// - Parameters:
///   - x: `[batch, length, hidden]`.
///   - indices: `[batch, length, k]` expert ids. Every id must be below the bank size.
///   - w13: `[experts, 2 * width, hidden]`, interleaved gate and up rows.
///   - w2: `[experts, hidden, width]`.
/// - Returns: `[batch, length, k, hidden]`, one expert output per selected id, before any weighting.
func sharpAIInklingMmExpertOutputs(
    _ x: MLXArray, indices: MLXArray, w13: MLXArray, w2: MLXArray, width: Int
) -> MLXArray {
    let xe = x.expandedDimensions(axis: -2).expandedDimensions(axis: -2)  // [B, T, 1, 1, H]
    let fused = MLX.gatherMM(xe, w13.swappedAxes(-1, -2), rhsIndices: indices)
        .squeezed(axis: -2)  // [B, T, K, 2 * width]
    let (gate, up) = sharpAIInklingMmSplitGateUp(fused, width: width)
    let hidden = (silu(gate) * up).expandedDimensions(axis: -2)  // [B, T, K, 1, width]
    return MLX.gatherMM(hidden, w2.swappedAxes(-1, -2), rhsIndices: indices)
        .squeezed(axis: -2)  // [B, T, K, hidden]
}

/// Routed expert bank (`mlp.experts.w13_weight`, `mlp.experts.w2_weight`).
final class SharpAIInklingMmExpertBank: Module {
    @ParameterInfo(key: "w13_weight") var w13Weight: MLXArray
    @ParameterInfo(key: "w2_weight") var w2Weight: MLXArray
    let width: Int

    init(hiddenSize: Int, width: Int, experts: Int) {
        _w13Weight.wrappedValue = MLXArray.zeros([experts, 2 * width, hiddenSize])
        _w2Weight.wrappedValue = MLXArray.zeros([experts, hiddenSize, width])
        self.width = width
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        sharpAIInklingMmExpertOutputs(x, indices: indices, w13: w13Weight, w2: w2Weight, width: width)
    }
}

/// Shared expert bank (`mlp.shared_experts.shared_w13_weight`, `mlp.shared_experts.shared_w2_weight`).
/// Every shared expert runs on every token.
final class SharpAIInklingMmSharedExpertBank: Module {
    @ParameterInfo(key: "shared_w13_weight") var sharedW13Weight: MLXArray
    @ParameterInfo(key: "shared_w2_weight") var sharedW2Weight: MLXArray
    let width: Int

    init(hiddenSize: Int, width: Int, sharedExperts: Int) {
        _sharedW13Weight.wrappedValue = MLXArray.zeros([sharedExperts, 2 * width, hiddenSize])
        _sharedW2Weight.wrappedValue = MLXArray.zeros([sharedExperts, hiddenSize, width])
        self.width = width
    }

    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        sharpAIInklingMmExpertOutputs(
            x, indices: indices, w13: sharedW13Weight, w2: sharedW2Weight, width: width)
    }
}

/// Sparse MoE block: routed experts plus shared experts, with the shared gammas from the router.
final class SharpAIInklingMmMoE: Module {
    @ModuleInfo(key: "gate") var gate: SharpAIInklingMmRouter
    @ModuleInfo(key: "experts") var experts: SharpAIInklingMmExpertBank
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIInklingMmSharedExpertBank
    let sharedCount: Int

    init(_ config: SharpAIInklingMmModelTextConfiguration) {
        sharedCount = config.sharedExperts
        _gate.wrappedValue = SharpAIInklingMmRouter(config)
        _experts.wrappedValue = SharpAIInklingMmExpertBank(
            hiddenSize: config.hiddenSize, width: config.moeIntermediateSize,
            experts: config.routedExperts)
        _sharedExperts.wrappedValue = SharpAIInklingMmSharedExpertBank(
            hiddenSize: config.hiddenSize, width: config.moeIntermediateSize,
            sharedExperts: config.sharedExperts)
    }

    /// `x` is `[batch, length, hidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routing = gate(x)
        let routed = (experts(x, indices: routing.indices) * routing.weights[.ellipsis, .newAxis])
            .sum(axis: -2)

        let sharedIndices = broadcast(
            MLXArray((0 ..< sharedCount).map { UInt32($0) }),
            to: [x.dim(0), x.dim(1), sharedCount])
        let shared = (sharedExperts(x, indices: sharedIndices) * routing.gammas[.ellipsis, .newAxis])
            .sum(axis: -2)
        return routed + shared
    }
}

/// One decoder layer. The attention and MLP sub-blocks each pass through a short convolution with
/// residual before their residual add, as transformers does.
final class SharpAIInklingMmDecoderLayer: Module {
    @ModuleInfo(key: "attn") var attn: SharpAIInklingMmAttention
    @ModuleInfo(key: "attn_norm") var attnNorm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "attn_sconv") var attnSconv: SharpAIInklingMmShortConv
    @ModuleInfo(key: "mlp_norm") var mlpNorm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "mlp_sconv") var mlpSconv: SharpAIInklingMmShortConv
    /// `SharpAIInklingMmDenseMLP` on dense layers and `SharpAIInklingMmMoE` on sparse layers.
    /// Both sit under the `mlp` key, so the checkpoint names resolve to one slot.
    @ModuleInfo(key: "mlp") var mlp: Module

    init(_ config: SharpAIInklingMmModelTextConfiguration, layer: Int) {
        _attn.wrappedValue = SharpAIInklingMmAttention(config, layer: layer)
        _attnNorm.wrappedValue = SharpAIInklingMmRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attnSconv.wrappedValue = SharpAIInklingMmShortConv(
            channels: config.hiddenSize, kernelSize: config.convKernelSize)
        _mlpNorm.wrappedValue = SharpAIInklingMmRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _mlpSconv.wrappedValue = SharpAIInklingMmShortConv(
            channels: config.hiddenSize, kernelSize: config.convKernelSize)
        if config.mlpLayerTypes[layer] == .dense {
            _mlp.wrappedValue = SharpAIInklingMmDenseMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        } else {
            _mlp.wrappedValue = SharpAIInklingMmMoE(config)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attnSconv(attn(attnNorm(x)))
        let normed = mlpNorm(h)
        let feedForward: MLXArray
        if let moe = mlp as? SharpAIInklingMmMoE {
            feedForward = moe(normed)
        } else if let dense = mlp as? SharpAIInklingMmDenseMLP {
            feedForward = dense(normed)
        } else {
            preconditionFailure("inkling_mm_model layer has neither a dense MLP nor a MoE block")
        }
        return h + mlpSconv(feedForward)
    }
}

/// Text stack: normed token embedding, decoder layers, final RMSNorm and the unembedding.
/// The checkpoint names are `model.llm.embed`, `model.llm.embed_norm`, `model.llm.layers`,
/// `model.llm.norm` and `model.llm.unembed`.
final class SharpAIInklingMmTextModel: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "layers") var layers: [SharpAIInklingMmDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIInklingMmRMSNorm
    @ModuleInfo(key: "unembed") var unembed: Linear

    init(_ config: SharpAIInklingMmModelTextConfiguration) {
        _embed.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedNorm.wrappedValue = SharpAIInklingMmRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            SharpAIInklingMmDecoderLayer(config, layer: $0)
        }
        _norm.wrappedValue = SharpAIInklingMmRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _unembed.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    /// Returns the final hidden states, before the muP division and the LM head.
    func hiddenStates(_ tokens: MLXArray) -> MLXArray {
        var h = embedNorm(embed(tokens))
        for layer in layers {
            h = layer(h)
        }
        return norm(h)
    }
}
