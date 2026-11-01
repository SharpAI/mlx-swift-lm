// SharpAI fork overlay: glm5_next text decoder, prefill forward. Not part of upstream mlx-swift-lm.
//
// Port of the text path of transformers `modeling_glm5_next.py`
// (src/transformers/models/glm5_next in huggingface/transformers). Implemented:
//   - manifold-constrained hyper-connections (mHC) with Sinkhorn-Knopp comb weights
//   - Kimi delta attention (KDA) linear layers, in the recurrent form for every length
//     (the same math as the reference chunked kernel, which is not ported)
//   - NoPE MLA attention gated by the DSA k-pool indexer (every layer uses the "full" indexer)
//   - dense MLP layers and sparse MoE layers (sigmoid router with grouped top-k)
//   - final stream mean, RMSNorm and lm_head
//
// Not implemented: KV and linear-state caches, padding masks (every token is valid), the
// vision tower, the MTP drafter, cross-layer top-k reuse ("shared" indexer types), and
// LanguageModel conformance. The creator in SharpAIGlm5NextRegistration.swift stays gated
// because a model without a cache cannot generate.
//
// Checkpoint key names follow the HF module tree. The expert tensors are stacked per projection
// (`mlp.switch_mlp.<proj>.weight`, the layout SharpAIGlm5NextSanitizer produces). The depthwise
// conv weight keeps the HF shape [channels, 1, kernel].

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Helpers

private func glmRequired<T>(_ value: T?, _ name: String) throws -> T {
    guard let value else {
        throw ModelFactoryError.invalidConfiguration("glm5_next: text_config.\(name) is missing")
    }
    return value
}

private func glmIota(_ count: Int) -> MLXArray {
    MLXArray((0 ..< count).map { Int32($0) })
}

private func glmClampMax(_ x: MLXArray, _ limit: Float) -> MLXArray {
    minimum(x, MLXArray(limit))
}

private func glmClampSymmetric(_ x: MLXArray, _ limit: Float) -> MLXArray {
    maximum(minimum(x, MLXArray(limit)), MLXArray(-limit))
}

private func glmL2Normalize(_ x: MLXArray) -> MLXArray {
    x * rsqrt((x * x).sum(axis: -1, keepDims: true) + Float(1e-6))
}

/// Unweighted RMS normalization used by the mHC stream projection.
private func glmUnweightedRMSNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(mean(square(x), axis: -1, keepDims: true) + eps)
}

// MARK: - Norms

final class SharpAIGlm5NextRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

/// LayerNorm with affine weight and bias (the indexer `k_norm`).
final class SharpAIGlm5NextLayerNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self._bias.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let centered = x - mean(x, axis: -1, keepDims: true)
        let variance = mean(square(centered), axis: -1, keepDims: true)
        return centered * rsqrt(variance + eps) * weight + bias
    }
}

// MARK: - Dense MLP and MoE

final class SharpAIGlm5NextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let swigluLimit: Float

    init(hidden: Int, intermediate: Int, swigluLimit: Float) {
        self.swigluLimit = swigluLimit
        self._gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        self._downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = glmClampMax(gateProj(x), swigluLimit)
        let up = glmClampSymmetric(upProj(x), swigluLimit)
        return downProj(silu(gate) * up)
    }
}

/// Stacked per-expert projection, weight shape [experts, out, in].
final class SharpAIGlm5NextStackedProjection: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(experts: Int, outputDimensions: Int, inputDimensions: Int) {
        self._weight.wrappedValue =
            MLXRandom.normal([experts, outputDimensions, inputDimensions]) * Float(0.02)
        super.init()
    }
}

final class SharpAIGlm5NextExperts: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SharpAIGlm5NextStackedProjection
    @ModuleInfo(key: "up_proj") var upProj: SharpAIGlm5NextStackedProjection
    @ModuleInfo(key: "down_proj") var downProj: SharpAIGlm5NextStackedProjection
    let experts: Int
    let swigluLimit: Float

    init(experts: Int, hidden: Int, intermediate: Int, swigluLimit: Float) {
        self.experts = experts
        self.swigluLimit = swigluLimit
        self._gateProj.wrappedValue = SharpAIGlm5NextStackedProjection(
            experts: experts, outputDimensions: intermediate, inputDimensions: hidden)
        self._upProj.wrappedValue = SharpAIGlm5NextStackedProjection(
            experts: experts, outputDimensions: intermediate, inputDimensions: hidden)
        self._downProj.wrappedValue = SharpAIGlm5NextStackedProjection(
            experts: experts, outputDimensions: hidden, inputDimensions: intermediate)
        super.init()
    }

    /// x: [N, H]. indices and weights: [N, K], the routed expert ids and their weights.
    /// Each expert runs on all tokens and is weighted by its routing mass. This is simple and
    /// exact, but not the sorted gather a production kernel would use.
    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        var total: MLXArray? = nil
        for e in 0 ..< experts {
            let routeWeight = (weights * (indices .== Int32(e)).asType(weights.dtype)).sum(axis: -1)
            let gate = glmClampMax(matmul(x, gateProj.weight[e].T), swigluLimit)
            let up = glmClampSymmetric(matmul(x, upProj.weight[e].T), swigluLimit)
            let out = matmul(silu(gate) * up, downProj.weight[e].T)
            let contribution = out * routeWeight.expandedDimensions(axis: -1)
            total = total.map { $0 + contribution } ?? contribution
        }
        guard let total else {
            preconditionFailure("glm5_next: at least one routed expert is required")
        }
        return total
    }
}

final class SharpAIGlm5NextRouter: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray
    let experts: Int
    let topK: Int
    let groups: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    init(
        hidden: Int, experts: Int, topK: Int, groups: Int, topkGroup: Int,
        routedScalingFactor: Float, normTopkProb: Bool
    ) {
        self.experts = experts
        self.topK = topK
        self.groups = groups
        self.topkGroup = topkGroup
        self.routedScalingFactor = routedScalingFactor
        self.normTopkProb = normTopkProb
        self._weight.wrappedValue = MLXRandom.normal([experts, hidden]) * Float(0.02)
        self._correctionBias.wrappedValue = MLXArray.zeros([experts])
        super.init()
    }

    /// x: [N, H]. Returns routing weights [N, K] and int32 expert indices [N, K].
    func callAsFunction(_ x: MLXArray) -> (weights: MLXArray, indices: MLXArray) {
        let n = x.dim(0)
        let logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        let scores = sigmoid(logits)
        let choice = scores + correctionBias.asType(.float32)

        // Group-limited selection: score each group by the sum of its top two choices.
        let perGroup = experts / groups
        let grouped = choice.reshaped([n, groups, perGroup])
        let topTwo = sorted(grouped, axis: -1)[0..., 0..., (perGroup - 2)...]
        let groupScores = topTwo.sum(axis: -1)
        let groupIdx = argPartition(-groupScores, kth: topkGroup - 1, axis: -1)[0..., ..<topkGroup]
        let groupMask = putAlong(
            MLXArray.zeros([n, groups]), groupIdx, values: MLXArray(Float(1)), axis: -1)
        let scoreMask = broadcast(
            groupMask.expandedDimensions(axis: -1), to: [n, groups, perGroup]
        ).reshaped([n, experts])
        let masked = MLX.where(scoreMask .> Float(0), choice, MLXArray(-Float.infinity))

        let indices = argPartition(-masked, kth: topK - 1, axis: -1)[0..., ..<topK].asType(.int32)
        var weights = takeAlong(scores, indices, axis: -1)
        if normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + Float(1e-20))
        }
        return (weights * routedScalingFactor, indices)
    }
}

final class SharpAIGlm5NextSparseMoE: Module {
    @ModuleInfo(key: "gate") var gate: SharpAIGlm5NextRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SharpAIGlm5NextExperts
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIGlm5NextMLP

    init(
        hidden: Int, experts: Int, topK: Int, groups: Int, topkGroup: Int,
        routedScalingFactor: Float, normTopkProb: Bool, moeIntermediate: Int,
        sharedExpertCount: Int, swigluLimit: Float
    ) {
        self._gate.wrappedValue = SharpAIGlm5NextRouter(
            hidden: hidden, experts: experts, topK: topK, groups: groups, topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb)
        self._switchMLP.wrappedValue = SharpAIGlm5NextExperts(
            experts: experts, hidden: hidden, intermediate: moeIntermediate, swigluLimit: swigluLimit)
        self._sharedExperts.wrappedValue = SharpAIGlm5NextMLP(
            hidden: hidden, intermediate: moeIntermediate * sharedExpertCount, swigluLimit: swigluLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped([-1, shape[shape.count - 1]])
        let routed = gate(flat)
        let out = switchMLP(flat, indices: routed.indices, weights: routed.weights)
        return out.reshaped(shape).asType(x.dtype) + sharedExperts(x)
    }
}

// MARK: - Manifold-constrained hyper-connections

final class SharpAIGlm5NextHyperConnection: Module {
    @ParameterInfo(key: "fn") var fn: MLXArray
    @ParameterInfo(key: "base") var base: MLXArray
    @ParameterInfo(key: "scale") var scale: MLXArray
    let streams: Int
    let sinkhornIterations: Int
    let eps: Float
    let rmsEps: Float

    init(hidden: Int, streams: Int, sinkhornIterations: Int, eps: Float, rmsEps: Float) {
        let outputs = (2 + streams) * streams
        self.streams = streams
        self.sinkhornIterations = sinkhornIterations
        self.eps = eps
        self.rmsEps = rmsEps
        self._fn.wrappedValue = MLXRandom.normal([outputs, streams * hidden]) * Float(0.02)
        self._base.wrappedValue = MLXArray.zeros([outputs])
        self._scale.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    /// streams: [B, S, N, D]. Returns post [B, S, N], comb [B, S, N, N] and the collapsed
    /// block input [B, S, D].
    func callAsFunction(_ streams: MLXArray) -> (post: MLXArray, comb: MLXArray, collapsed: MLXArray) {
        let batch = streams.dim(0)
        let seq = streams.dim(1)
        let n = self.streams

        let flat = glmUnweightedRMSNorm(
            streams.reshaped([batch, seq, -1]).asType(.float32), eps: rmsEps)
        let mixed = matmul(flat, fn.asType(.float32).T)
        let preW = mixed[0..., 0..., ..<n]
        let postW = mixed[0..., 0..., n ..< (2 * n)]
        let combW = mixed[0..., 0..., (2 * n)...].reshaped([batch, seq, n, n])

        let base = self.base.asType(.float32)
        let preB = base[..<n]
        let postB = base[n ..< (2 * n)]
        let combB = base[(2 * n)...].reshaped([n, n])
        let s = scale.asType(.float32)

        let pre = sigmoid(preW * s[0] + preB) + eps
        let post = 2 * sigmoid(postW * s[1] + postB)
        var comb = softmax(combW * s[2] + combB, axis: -1) + eps

        // Sinkhorn-Knopp: push comb towards a doubly stochastic matrix.
        comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        for _ in 0 ..< max(sinkhornIterations - 1, 0) {
            comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
            comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
        }

        let collapsed = (pre.expandedDimensions(axis: -1) * streams.asType(.float32)).sum(axis: 2)
        return (post, comb, collapsed.asType(streams.dtype))
    }
}

// MARK: - Kimi delta attention (linear layers)

/// Depthwise causal conv weight, stored with the HF shape [channels, 1, kernel].
final class SharpAIGlm5NextDepthwiseConv: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(channels: Int, kernel: Int) {
        self._weight.wrappedValue = MLXRandom.normal([channels, 1, kernel]) * Float(0.1)
        super.init()
    }
}

final class SharpAIGlm5NextForgetGate: Module {
    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    let heads: Int
    let headDim: Int
    let lowerBound: Float

    init(hidden: Int, heads: Int, headDim: Int, lowerBound: Float) {
        self.heads = heads
        self.headDim = headDim
        self.lowerBound = lowerBound
        self._fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        self._fBProj.wrappedValue = Linear(headDim, heads * headDim, bias: false)
        self._dtBias.wrappedValue = MLXArray.zeros([heads * headDim])
        self._aLog.wrappedValue = MLXArray.zeros([heads])
        super.init()
    }

    /// Returns the per-channel log decay [B, S, H, D], bounded below by `lowerBound`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seq = x.dim(1)
        let raw = fBProj(fAProj(x)).asType(.float32) + dtBias.asType(.float32)
        let g = raw.reshaped([batch, seq, heads, headDim])
        let decay = exp(aLog.asType(.float32)).reshaped([1, 1, heads, 1])
        return lowerBound * sigmoid(decay * g)
    }
}

final class SharpAIGlm5NextLinearAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "conv1d") var conv1d: SharpAIGlm5NextDepthwiseConv
    @ModuleInfo(key: "forget_gate") var forgetGate: SharpAIGlm5NextForgetGate
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear
    @ModuleInfo(key: "o_norm") var oNorm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let heads: Int
    let headDim: Int
    let qkvDim: Int
    let kernel: Int
    let scale: Float

    init(hidden: Int, heads: Int, headDim: Int, kernel: Int, lowerBound: Float, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        self.qkvDim = heads * headDim
        self.kernel = kernel
        self.scale = 1 / Float(headDim).squareRoot()
        let qkv = heads * headDim
        self._qProj.wrappedValue = Linear(hidden, qkv, bias: false)
        self._kProj.wrappedValue = Linear(hidden, qkv, bias: false)
        self._vProj.wrappedValue = Linear(hidden, qkv, bias: false)
        self._conv1d.wrappedValue = SharpAIGlm5NextDepthwiseConv(channels: 3 * qkv, kernel: kernel)
        self._forgetGate.wrappedValue = SharpAIGlm5NextForgetGate(
            hidden: hidden, heads: heads, headDim: headDim, lowerBound: lowerBound)
        self._bProj.wrappedValue = Linear(hidden, heads, bias: false)
        self._gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        self._gBProj.wrappedValue = Linear(headDim, qkv, bias: false)
        self._oNorm.wrappedValue = SharpAIGlm5NextRMSNorm(dimensions: headDim, eps: eps)
        self._oProj.wrappedValue = Linear(qkv, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seq = x.dim(1)

        let mixed = concatenated([qProj(x), kProj(x), vProj(x)], axis: -1)
        let convolved = causalDepthwiseSiLU(mixed)
        let query = convolved[0..., 0..., ..<qkvDim].reshaped([batch, seq, heads, headDim])
        let key = convolved[0..., 0..., qkvDim ..< (2 * qkvDim)].reshaped([batch, seq, heads, headDim])
        let value = convolved[0..., 0..., (2 * qkvDim)...].reshaped([batch, seq, heads, headDim])

        let q = glmL2Normalize(query.asType(.float32)) * scale
        let k = glmL2Normalize(key.asType(.float32))
        let v = value.asType(.float32)
        let g = forgetGate(x)
        let beta = sigmoid(bProj(x)).asType(.float32)

        let core = recurrentDeltaAttention(q: q, k: k, v: v, g: g, beta: beta)
        let gate = gBProj(gAProj(x)).reshaped([batch, seq, heads, headDim]).asType(.float32)
        let normed = oNorm(core.asType(.float32)) * sigmoid(gate)
        return oProj(normed.asType(x.dtype).reshaped([batch, seq, qkvDim]))
    }

    /// Causal depthwise conv over the sequence axis, then SiLU. x: [B, S, C].
    private func causalDepthwiseSiLU(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seq = x.dim(1)
        let channels = x.dim(2)
        let weight = conv1d.weight.reshaped([channels, kernel]).asType(x.dtype)
        let padded = concatenated(
            [MLXArray.zeros([batch, kernel - 1, channels]).asType(x.dtype), x], axis: 1)
        var acc = MLXArray.zeros([batch, seq, channels]).asType(x.dtype)
        for j in 0 ..< kernel {
            acc = acc + padded[0..., j ..< (j + seq), 0...] * weight[0..., j]
        }
        return silu(acc)
    }

    /// Recurrent Kimi delta rule (FLA `fused_recurrent_kda` math). q, k, v: [B, S, H, D] float32,
    /// g: [B, S, H, D] log decay, beta: [B, S, H]. Returns [B, S, H, D] float32.
    private func recurrentDeltaAttention(
        q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray
    ) -> MLXArray {
        let batch = q.dim(0)
        let seq = q.dim(1)
        var state = MLXArray.zeros([batch, heads, headDim, headDim])
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(seq)
        for t in 0 ..< seq {
            let qt = q[0..., t].expandedDimensions(axis: -1)
            let kt = k[0..., t].expandedDimensions(axis: -1)
            let vt = v[0..., t]
            let decay = exp(g[0..., t]).expandedDimensions(axis: -1)
            let bt = beta[0..., t].expandedDimensions(axis: -1)

            state = state * decay
            let memory = (state * kt).sum(axis: -2)
            let delta = (vt - memory) * bt
            state = state + kt * delta.expandedDimensions(axis: -2)
            outputs.append((state * qt).sum(axis: -2))
        }
        return stacked(outputs, axis: 1)
    }
}

// MARK: - NoPE MLA attention with DSA k-pool indexer

final class SharpAIGlm5NextIndexer: Module {
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIGlm5NextLayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ParameterInfo(key: "index_kpool_compress_ape") var compressAPE: MLXArray
    @ParameterInfo(key: "index_kpool_compress_gate") var compressGate: MLXArray

    let heads: Int
    let headDim: Int
    let topk: Int
    let kpool: Int
    let alwaysSelectTail: Bool
    let softmaxScale: Float

    init(
        hidden: Int, qLoraRank: Int, heads: Int, headDim: Int, topk: Int, kpool: Int,
        alwaysSelectTail: Bool
    ) {
        self.heads = heads
        self.headDim = headDim
        self.topk = topk
        self.kpool = kpool
        self.alwaysSelectTail = alwaysSelectTail
        self.softmaxScale = 1 / Float(headDim).squareRoot()
        self._wqB.wrappedValue = Linear(qLoraRank, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(hidden, headDim, bias: false)
        self._kNorm.wrappedValue = SharpAIGlm5NextLayerNorm(dimensions: headDim, eps: 1e-6)
        self._weightsProj.wrappedValue = Linear(hidden, heads, bias: false)
        self._compressAPE.wrappedValue = MLXArray.zeros([kpool, headDim])
        self._compressGate.wrappedValue = MLXArray.ones([headDim, hidden])
        super.init()
    }

    /// x: [B, S, hidden], qResid: [B, S, q_lora_rank]. Returns int32 token indices [B, S, W],
    /// -1 marking unused slots. Every token is valid, so the causal rule is the only mask.
    /// W is topk, plus kpool - 1 when the incomplete tail is always selected.
    func callAsFunction(_ x: MLXArray, qResid: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seq = x.dim(1)
        let pools = seq / kpool
        let tailWidth = alwaysSelectTail ? kpool - 1 : 0
        let outputWidth = topk + tailWidth
        let selectK = min(topk / kpool, pools)

        var pieces: [MLXArray] = []

        if pools > 0 && selectK > 0 {
            let q = wqB(qResid).reshaped([batch, seq, heads, headDim]).asType(.float32)
            let k = kNorm(wk(x)).asType(.float32)
            let gateScores = matmul(x, compressGate.T).asType(.float32)

            // Complete pools only: a weighted average of each pool's keys.
            let span = pools * kpool
            let groupedKeys = k[0..., ..<span].reshaped([batch, pools, kpool, headDim])
            let groupedGates = gateScores[0..., ..<span].reshaped([batch, pools, kpool, headDim])
            let probabilities = softmax(groupedGates + compressAPE.asType(.float32), axis: 2)
            let poolKeys = (probabilities * groupedKeys).sum(axis: 2)

            // Per query head score against each pool: [B, S, H, P].
            var scores = matmul(q, poolKeys.transposed(0, 2, 1).expandedDimensions(axis: 1))
            scores = relu(scores * softmaxScale)
            let headWeights = weightsProj(x).asType(.float32) * (1 / Float(heads).squareRoot())
            let indexScores = (scores * headWeights.expandedDimensions(axis: -1)).sum(axis: 2)

            // A pool is a candidate for query q only if its last token is causally visible.
            var candidate = [Int32](repeating: 0, count: seq * pools)
            for qpos in 0 ..< seq {
                for p in 0 ..< pools where qpos >= p * kpool + kpool - 1 {
                    candidate[qpos * pools + p] = 1
                }
            }
            let visible = broadcast(
                MLXArray(candidate, [1, seq, pools]).asType(.bool), to: [batch, seq, pools])
            let masked = MLX.where(visible, indexScores, MLXArray(-Float.greatestFiniteMagnitude))

            let selected = argPartition(-masked, kth: selectK - 1, axis: -1)[0..., 0..., ..<selectK]
                .asType(.int32)
            let selectedValid = takeAlong(visible, selected, axis: -1)
            let tokenBase = (selected * Int32(kpool)).expandedDimensions(axis: -1)
            let tokens = tokenBase + glmIota(kpool)
            let guarded = MLX.where(
                selectedValid.expandedDimensions(axis: -1), tokens, MLXArray(Int32(-1)))
            pieces.append(guarded.reshaped([batch, seq, selectK * kpool]))
        }

        if tailWidth > 0 {
            // The incomplete tail of each query: the last (visible count % kpool) tokens.
            var tailRows = [Int32](repeating: -1, count: seq * tailWidth)
            for qpos in 0 ..< seq {
                let visibleCount = qpos + 1
                let tailCount = visibleCount % kpool
                let tailStart = visibleCount - tailCount
                for j in 0 ..< tailCount where tailStart + j < seq {
                    tailRows[qpos * tailWidth + j] = Int32(tailStart + j)
                }
            }
            pieces.append(broadcast(MLXArray(tailRows, [1, seq, tailWidth]), to: [batch, seq, tailWidth]))
        }

        let width = pieces.reduce(0) { $0 + $1.dim(-1) }
        if width < outputWidth {
            pieces.append(broadcast(MLXArray(Int32(-1)), to: [batch, seq, outputWidth - width]))
        }
        return concatenated(pieces, axis: -1)
    }
}

final class SharpAIGlm5NextIndexedAttention: Module {
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayernorm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayernorm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: SharpAIGlm5NextIndexer

    let heads: Int
    let nopeDim: Int
    let valueDim: Int
    let scale: Float

    init(
        hidden: Int, heads: Int, qLoraRank: Int, kvLoraRank: Int, nopeDim: Int, valueDim: Int,
        attentionBias: Bool, rmsEps: Float, indexer: SharpAIGlm5NextIndexer
    ) {
        self.heads = heads
        self.nopeDim = nopeDim
        self.valueDim = valueDim
        self.scale = 1 / Float(nopeDim).squareRoot()
        self._qAProj.wrappedValue = Linear(hidden, qLoraRank, bias: attentionBias)
        self._qALayernorm.wrappedValue = SharpAIGlm5NextRMSNorm(dimensions: qLoraRank, eps: rmsEps)
        self._qBProj.wrappedValue = Linear(qLoraRank, heads * nopeDim, bias: false)
        self._kvAProj.wrappedValue = Linear(hidden, kvLoraRank, bias: attentionBias)
        self._kvALayernorm.wrappedValue = SharpAIGlm5NextRMSNorm(dimensions: kvLoraRank, eps: rmsEps)
        self._kvBProj.wrappedValue = Linear(kvLoraRank, heads * (nopeDim + valueDim), bias: false)
        self._oProj.wrappedValue = Linear(heads * valueDim, hidden, bias: attentionBias)
        self._indexer.wrappedValue = indexer
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let seq = x.dim(1)
        let qResid = qALayernorm(qAProj(x))
        let queries = qBProj(qResid).reshaped([batch, seq, heads, nopeDim]).transposed(0, 2, 1, 3)

        let kvPass = kvALayernorm(kvAProj(x))
        let expanded = kvBProj(kvPass)
            .reshaped([batch, seq, heads, nopeDim + valueDim])
            .transposed(0, 2, 1, 3)
        let keys = expanded[0..., 0..., 0..., ..<nopeDim]
        let values = expanded[0..., 0..., 0..., nopeDim...]

        let topk = indexer(x, qResid: qResid)
        let mask = Self.mask(fromTopK: topk, seq: seq)

        var scores = matmul(queries.asType(.float32), keys.asType(.float32).transposed(0, 1, 3, 2))
        scores = scores * scale
        scores = MLX.where(
            mask.expandedDimensions(axis: 1), scores, MLXArray(-Float.greatestFiniteMagnitude))
        let probs = softmax(scores, axis: -1).asType(x.dtype)
        let out = matmul(probs, values)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, seq, heads * valueDim])
        return oProj(out)
    }

    /// Visibility mask [B, S, S] from the indexer's selected indices. -1 entries are dropped.
    private static func mask(fromTopK topk: MLXArray, seq: Int) -> MLXArray {
        let batch = topk.dim(0)
        let slot = MLX.where(topk .>= Int32(0), topk, MLXArray(Int32(seq)))
        let zeros = MLXArray(
            [Int32](repeating: 0, count: batch * seq * (seq + 1)), [batch, seq, seq + 1])
        let counts = putAlong(zeros, slot, values: MLXArray(Int32(1)), axis: -1)
        return counts[0..., 0..., ..<seq] .> Int32(0)
    }
}

// MARK: - Decoder layer and model

final class SharpAIGlm5NextDecoderLayer: Module {
    // HF names both attention kinds `self_attn` and both FFN kinds `mlp`. MLXNN keeps one entry
    // per key, so each key has a single property holding whichever kind the layer uses.
    @ModuleInfo(key: "self_attn") var selfAttn: Module?
    @ModuleInfo(key: "mlp") var mlp: Module?
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: SharpAIGlm5NextHyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: SharpAIGlm5NextHyperConnection

    init(
        linearAttention: SharpAIGlm5NextLinearAttention?,
        indexedAttention: SharpAIGlm5NextIndexedAttention?,
        denseMLP: SharpAIGlm5NextMLP?,
        sparseMoE: SharpAIGlm5NextSparseMoE?,
        hidden: Int, rmsEps: Float, hcStreams: Int, hcSinkhornIterations: Int, hcEps: Float
    ) {
        self._selfAttn.wrappedValue = linearAttention ?? indexedAttention
        self._mlp.wrappedValue = denseMLP ?? sparseMoE
        self._inputLayernorm.wrappedValue = SharpAIGlm5NextRMSNorm(dimensions: hidden, eps: rmsEps)
        self._postAttentionLayernorm.wrappedValue = SharpAIGlm5NextRMSNorm(
            dimensions: hidden, eps: rmsEps)
        self._attnHC.wrappedValue = SharpAIGlm5NextHyperConnection(
            hidden: hidden, streams: hcStreams, sinkhornIterations: hcSinkhornIterations,
            eps: hcEps, rmsEps: rmsEps)
        self._ffnHC.wrappedValue = SharpAIGlm5NextHyperConnection(
            hidden: hidden, streams: hcStreams, sinkhornIterations: hcSinkhornIterations,
            eps: hcEps, rmsEps: rmsEps)
        super.init()
    }

    /// streams: [B, S, N, D]. Returns [B, S, N, D].
    func callAsFunction(_ streams: MLXArray) -> MLXArray {
        let dtype = streams.dtype

        let attnIn = attnHC(streams)
        var attnOut = inputLayernorm(attnIn.collapsed)
        if let linearAttention = selfAttn as? SharpAIGlm5NextLinearAttention {
            attnOut = linearAttention(attnOut)
        } else if let indexedAttention = selfAttn as? SharpAIGlm5NextIndexedAttention {
            attnOut = indexedAttention(attnOut)
        }
        let afterAttn = Self.combine(
            attnOut, post: attnIn.post, comb: attnIn.comb, residual: streams, dtype: dtype)

        let ffnIn = ffnHC(afterAttn)
        var ffnOut = postAttentionLayernorm(ffnIn.collapsed)
        if let denseMLP = mlp as? SharpAIGlm5NextMLP {
            ffnOut = denseMLP(ffnOut)
        } else if let sparseMoE = mlp as? SharpAIGlm5NextSparseMoE {
            ffnOut = sparseMoE(ffnOut)
        }
        return Self.combine(
            ffnOut, post: ffnIn.post, comb: ffnIn.comb, residual: afterAttn, dtype: dtype)
    }

    private static func combine(
        _ block: MLXArray, post: MLXArray, comb: MLXArray, residual: MLXArray, dtype: DType
    ) -> MLXArray {
        post.asType(dtype).expandedDimensions(axis: -1) * block.expandedDimensions(axis: -2)
            + matmul(comb.asType(dtype).transposed(0, 1, 3, 2), residual)
    }
}

/// glm5_next text decoder. Prefill forward only (see the file header for what is missing).
public final class SharpAIGlm5NextTextModel: Module {
    let hcStreams: Int

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [SharpAIGlm5NextDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIGlm5NextRMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ configuration: SharpAIGlm5NextTextConfiguration) throws {
        let hidden = try glmRequired(configuration.hiddenSize, "hidden_size")
        let vocab = try glmRequired(configuration.vocabSize, "vocab_size")
        let layerCount = try glmRequired(configuration.numHiddenLayers, "num_hidden_layers")
        let heads = try glmRequired(configuration.numAttentionHeads, "num_attention_heads")
        let rmsEps = configuration.rmsNormEps ?? 1e-5
        let attentionBias = configuration.attentionBias ?? false
        let swigluLimit = configuration.swigluLimit ?? 10.0

        let qLoraRank = try glmRequired(configuration.qLoraRank, "q_lora_rank")
        let kvLoraRank = try glmRequired(configuration.kvLoraRank, "kv_lora_rank")
        let nopeDim = try glmRequired(configuration.qkNopeHeadDim, "qk_nope_head_dim")
        let valueDim = try glmRequired(configuration.vHeadDim, "v_head_dim")
        let ropeDim = configuration.qkRopeHeadDim ?? 0
        guard ropeDim == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: qk_rope_head_dim must be 0 for the NoPE MLA path, got \(ropeDim)")
        }

        let hcStreams = try glmRequired(configuration.hcMult, "hc_mult")
        let hcIterations = try glmRequired(configuration.hcSinkhornIters, "hc_sinkhorn_iters")
        let hcEps = configuration.hcEps ?? 1e-6

        let linear = configuration.linearAttnConfig
        let linearHeads = try glmRequired(linear?.numHeads, "linear_attn_config.num_heads")
        let linearHeadDim = try glmRequired(linear?.headDim, "linear_attn_config.head_dim")
        let convKernel = linear?.shortConvKernelSize ?? 4
        let lowerBound = linear?.gateLowerBound ?? -5.0

        let indexHeads = try glmRequired(configuration.indexNHeads, "index_n_heads")
        let indexHeadDim = try glmRequired(configuration.indexHeadDim, "index_head_dim")
        let indexTopk = try glmRequired(configuration.indexTopk, "index_topk")
        let indexKpool = try glmRequired(configuration.indexKpool, "index_kpool")
        let alwaysSelectTail = configuration.indexKpoolAlwaysSelectTail ?? true
        guard indexKpool >= 1 else {
            throw ModelFactoryError.invalidConfiguration("glm5_next: index_kpool must be >= 1")
        }

        let denseIntermediate = try glmRequired(configuration.intermediateSize, "intermediate_size")
        let experts = try glmRequired(configuration.nRoutedExperts, "n_routed_experts")
        let expertsPerToken = try glmRequired(configuration.numExpertsPerTok, "num_experts_per_tok")
        let moeIntermediate = try glmRequired(configuration.moeIntermediateSize, "moe_intermediate_size")
        let sharedCount = try glmRequired(configuration.nSharedExperts, "n_shared_experts")
        let groups = configuration.nGroup ?? 1
        let topkGroup = configuration.topkGroup ?? 1
        let routedScale = configuration.routedScalingFactor ?? 1
        let normTopk = configuration.normTopkProb ?? true
        guard experts >= 1, groups >= 1, experts % groups == 0, experts / groups >= 2 else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: n_routed_experts must split into n_group groups of at least 2 experts")
        }
        guard expertsPerToken >= 1, expertsPerToken <= experts, topkGroup >= 1, topkGroup <= groups,
            sharedCount >= 1
        else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: expert routing counts are out of range")
        }
        guard (configuration.scoringFunc ?? "sigmoid") == "sigmoid" else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: only sigmoid scoring is implemented")
        }

        let layerTypes = configuration.layerTypes
            ?? (0 ..< layerCount).map { index in
                (linear?.kdaLayers ?? []).contains(index) ? "linear_attention" : "deepseek_sparse_attention"
            }
        let indexerTypes = configuration.indexerTypes ?? Array(repeating: "full", count: layerCount)
        let mlpTypes = configuration.mlpLayerTypes
            ?? (0 ..< layerCount).map { index in
                index < (configuration.firstKDenseReplace ?? 0) ? "dense" : "sparse"
            }
        guard layerTypes.count == layerCount, mlpTypes.count == layerCount,
            indexerTypes.count == layerCount
        else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: layer_types, mlp_layer_types and indexer_types must have num_hidden_layers entries")
        }
        guard indexerTypes.allSatisfy({ $0 == "full" }) else {
            throw ModelFactoryError.invalidConfiguration(
                "glm5_next: cross-layer top-k sharing (indexer_types \"shared\") is not implemented")
        }

        var built: [SharpAIGlm5NextDecoderLayer] = []
        built.reserveCapacity(layerCount)
        for index in 0 ..< layerCount {
            var linearLayer: SharpAIGlm5NextLinearAttention? = nil
            var indexedLayer: SharpAIGlm5NextIndexedAttention? = nil
            switch layerTypes[index] {
            case "linear_attention":
                linearLayer = SharpAIGlm5NextLinearAttention(
                    hidden: hidden, heads: linearHeads, headDim: linearHeadDim,
                    kernel: convKernel, lowerBound: lowerBound, eps: rmsEps)
            case "deepseek_sparse_attention":
                let indexer = SharpAIGlm5NextIndexer(
                    hidden: hidden, qLoraRank: qLoraRank, heads: indexHeads, headDim: indexHeadDim,
                    topk: indexTopk, kpool: indexKpool, alwaysSelectTail: alwaysSelectTail)
                indexedLayer = SharpAIGlm5NextIndexedAttention(
                    hidden: hidden, heads: heads, qLoraRank: qLoraRank, kvLoraRank: kvLoraRank,
                    nopeDim: nopeDim, valueDim: valueDim, attentionBias: attentionBias,
                    rmsEps: rmsEps, indexer: indexer)
            default:
                throw ModelFactoryError.invalidConfiguration(
                    "glm5_next: unsupported layer type \(layerTypes[index])")
            }

            var denseLayer: SharpAIGlm5NextMLP? = nil
            var sparseLayer: SharpAIGlm5NextSparseMoE? = nil
            switch mlpTypes[index] {
            case "dense":
                denseLayer = SharpAIGlm5NextMLP(
                    hidden: hidden, intermediate: denseIntermediate, swigluLimit: swigluLimit)
            case "sparse":
                sparseLayer = SharpAIGlm5NextSparseMoE(
                    hidden: hidden, experts: experts, topK: expertsPerToken, groups: groups,
                    topkGroup: topkGroup, routedScalingFactor: routedScale, normTopkProb: normTopk,
                    moeIntermediate: moeIntermediate, sharedExpertCount: sharedCount,
                    swigluLimit: swigluLimit)
            default:
                throw ModelFactoryError.invalidConfiguration(
                    "glm5_next: unsupported mlp layer type \(mlpTypes[index])")
            }

            built.append(
                SharpAIGlm5NextDecoderLayer(
                    linearAttention: linearLayer, indexedAttention: indexedLayer,
                    denseMLP: denseLayer, sparseMoE: sparseLayer, hidden: hidden, rmsEps: rmsEps,
                    hcStreams: hcStreams, hcSinkhornIterations: hcIterations, hcEps: hcEps))
        }

        self.hcStreams = hcStreams
        self._embedTokens.wrappedValue = Embedding(embeddingCount: vocab, dimensions: hidden)
        self._layers.wrappedValue = built
        self._norm.wrappedValue = SharpAIGlm5NextRMSNorm(dimensions: hidden, eps: rmsEps)
        self._lmHead.wrappedValue = Linear(hidden, vocab, bias: false)
        super.init()
    }

    /// inputIds: [B, S] integer token ids. Returns logits [B, S, vocab].
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let embedded = embedTokens(inputIds).expandedDimensions(axis: 2)
        var streams = concatenated(Array(repeating: embedded, count: hcStreams), axis: 2)
        for layer in layers {
            streams = layer(streams)
        }
        let pooled = streams.mean(axis: 2)
        return lmHead(norm(pooled))
    }
}
