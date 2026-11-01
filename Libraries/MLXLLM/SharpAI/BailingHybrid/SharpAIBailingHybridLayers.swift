// SharpAI fork overlay: bailing_hybrid layers. Not part of upstream mlx-swift-lm.
//
// Port of mlx_lm/models/bailing_hybrid.py (ml-explore/mlx-lm, PR #1227):
//   - recurrentGLA: the lightning recurrence h_t = exp(g) h_{t-1} + k_t^T v_t, y_t = q_t h_t,
//     in its single-step and chunked forms (SharpAIBailingHybridRecurrence).
//   - MultiLatentAttention for global layers, LightningAttention for the others.
//   - Sigmoid group-limited router (expert bias steers selection only), SwiGLU experts, shared experts.
// Differences from the Python reference:
//   - The recurrence runs in float32 throughout (the Python reference casts k/v to float32 in the chunked path).
//   - YaRN mscale is not applied to the MLA softmax scale. The released config has rope_scaling null.
//   - Linear-attention layers run without a padding mask, so padded batches are not supported yet.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Lightning recurrence

public enum SharpAIBailingHybridRecurrence {

    /// Lightning-attention recurrence ``h_t = h_{t-1} exp(g) + k_t^T v_t``, ``y_t = (scale q_t) h_t``,
    /// with ``g`` a per-head decay constant over time (shape `[H]`).
    ///
    /// For L == 1 this is one step. For L > 1 it runs in chunks of `chunk` steps: each chunk is one
    /// decay-weighted causal attention plus one state carry. `mask` is an optional `[B, L]`
    /// validity mask: padded positions add nothing to the state and do not decay it.
    /// - Returns: `[B, H, L, V]` output (in `q`'s dtype) and the float32 state `[B, H, K, V]`.
    public static func recurrentGLA(
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        g: MLXArray,
        scale: Float,
        state: MLXArray? = nil,
        mask: MLXArray? = nil,
        chunk: Int = 128
    ) -> (output: MLXArray, state: MLXArray) {
        let B = q.dim(0)
        let H = q.dim(1)
        let L = q.dim(2)
        let K = q.dim(3)
        let V = v.dim(3)
        let outDType = q.dtype

        let q32 = q.asType(.float32) * scale
        let k32 = k.asType(.float32)
        let v32 = v.asType(.float32)
        let gh = g.asType(.float32).reshaped(1, H, 1, 1)
        var h = state?.asType(.float32) ?? MLXArray.zeros([B, H, K, V], dtype: .float32)

        if L == 1 {
            let expG = MLX.exp(gh)
            let newState = h * expG + k32.transposed(0, 1, 3, 2).matmul(v32)
            let out = q32.matmul(newState).asType(outDType)
            guard let mask else { return (out, newState) }
            let keep = mask[0..., 0].asType(.bool).reshaped(B, 1, 1, 1)
            return (
                MLX.where(keep, out, MLXArray(Float(0)).asType(outDType)),
                MLX.where(keep, newState, h)
            )
        }

        let C = min(chunk, L, 128)
        let nChunks = (L + C - 1) / C
        let total = nChunks * C
        let pad = total - L

        // Rank counts valid positions, so padding neither contributes nor decays the state.
        var m: MLXArray
        if let mask {
            m = mask.asType(.float32)
            if pad > 0 { m = padded(m, widths: [.init((0, 0)), .init((0, pad))]) }
        } else {
            m = MLXArray((0 ..< total).map { $0 < L ? Float(1) : Float(0) }).reshaped(1, total)
        }

        var qp = q32
        var kp = k32
        var vp = v32
        if pad > 0 {
            qp = padded(q32, widths: [.init((0, 0)), .init((0, 0)), .init((0, pad)), .init((0, 0))])
            kp = padded(k32, widths: [.init((0, 0)), .init((0, 0)), .init((0, pad)), .init((0, 0))])
            vp = padded(v32, widths: [.init((0, 0)), .init((0, 0)), .init((0, pad)), .init((0, 0))])
        }

        let qc = qp.reshaped(B, H, nChunks, C, K)
        let kc = kp.reshaped(B, H, nChunks, C, K)
        let vc = vp.reshaped(B, H, nChunks, C, V)
        let mc = m.reshaped(m.dim(0), 1, nChunks, C)

        let rank = MLX.cumsum(mc, axis: -1)
        let chunkTotal = rank[0..., 0..., 0..., (C - 1) ..< C]
        // Centring the rank splits exp(g (r_i - r_j)) over i and j to keep float32 in range.
        let centre = 0.5 * Float(C)
        let e = gh * (rank - centre)
        let eq = MLX.exp(e) * mc
        let ek = MLX.exp(-e) * mc
        let qs = qc * eq.expandedDimensions(axis: -1)
        let ks = kc * ek.expandedDimensions(axis: -1)

        // tril selects, so the strict upper triangle (which may overflow) is dropped, not multiplied.
        let scores = MLX.tril(qs.matmul(ks.swappedAxes(-1, -2)))
        let intra = scores.matmul(vc)
        let carry = ks.swappedAxes(-1, -2).matmul(vc)
        let decay = MLX.exp(gh * (chunkTotal - centre)).expandedDimensions(axis: -1)
        let stateDecay = MLX.exp(gh * chunkTotal).expandedDimensions(axis: -1)

        var cross: [MLXArray] = []
        cross.reserveCapacity(nChunks)
        for i in 0 ..< nChunks {
            cross.append(qs[0..., 0..., i].matmul(h))
            h = stateDecay[0..., 0..., i] * h + carry[0..., 0..., i] * decay[0..., 0..., i]
        }
        let crossAll = MLX.stacked(cross, axis: 2)
        let crossScale = MLX.exp(gh * centre).reshaped(1, H, 1, 1, 1)

        let y = intra + crossAll * crossScale
        let out = y.reshaped(B, H, total, V)[0..., 0..., ..<L, 0...].asType(outDType)
        return (out, h)
    }
}

// MARK: - Building blocks

/// Per-head linear map with weight `[numHeads, outputDims, inputDims]`, as in mla.MultiLinear.
final class SharpAIBailingHybridMultiLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, numHeads: Int) {
        let bound = sqrt(1.0 / Float(inputDims))
        _weight.wrappedValue = MLXRandom.uniform(
            low: -bound, high: bound, [numHeads, outputDims, inputDims])
        super.init()
    }

    /// `transpose: true` computes x @ weight^T (input dims -> output dims).
    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        transpose ? x.matmul(weight.swappedAxes(-1, -2)) : x.matmul(weight)
    }
}

/// GroupRMSNorm: RMS-normalise each group of channels, then apply a learned per-channel weight.
final class SharpAIBailingHybridGroupRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let groups: Int
    let eps: Float

    init(dimensions: Int, eps: Float, groups: Int) {
        self.groups = groups
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let grouped = x.reshaped(B, L, groups, -1)
        let normed = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
            .reshaped(B, L, -1)
        return weight * normed
    }
}

final class SharpAIBailingHybridMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, bias: Bool) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Global attention (MLA)

final class SharpAIBailingHybridMultiLatentAttention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qkHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayernorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayernorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: SharpAIBailingHybridMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: SharpAIBailingHybridMultiLinear
    @ModuleInfo(key: "dense") var dense: Linear

    init(_ c: SharpAIBailingHybridConfiguration) {
        let hidden = c.hiddenSize
        let heads = c.attentionHeads
        let nope = c.qkNopeHeadDim ?? 0
        let ropeDim = c.qkRopeHeadDim ?? 0
        let qkDim = nope + ropeDim
        let kvLora = c.kvLoraRank ?? 0
        let vDim = c.vHeadDim ?? 0

        numHeads = heads
        qkNopeHeadDim = nope
        qkRopeHeadDim = ropeDim
        qkHeadDim = qkDim
        kvLoraRank = kvLora
        vHeadDim = vDim
        scale = pow(Float(qkDim), -0.5)

        if let qLora = c.qLoraRank {
            _qAProj.wrappedValue = Linear(hidden, qLora, bias: c.useQKVBias)
            _qALayernorm.wrappedValue = RMSNorm(dimensions: qLora, eps: c.rmsNormEps)
            _qBProj.wrappedValue = Linear(qLora, heads * qkDim, bias: false)
            _qProj.wrappedValue = nil
        } else {
            _qAProj.wrappedValue = nil
            _qALayernorm.wrappedValue = nil
            _qBProj.wrappedValue = nil
            _qProj.wrappedValue = Linear(hidden, heads * qkDim, bias: false)
        }

        _kvAProjWithMQA.wrappedValue = Linear(hidden, kvLora + ropeDim, bias: c.useQKVBias)
        _kvALayernorm.wrappedValue = RMSNorm(dimensions: kvLora, eps: c.rmsNormEps)
        _embedQ.wrappedValue = SharpAIBailingHybridMultiLinear(
            inputDims: nope, outputDims: kvLora, numHeads: heads)
        _unembedOut.wrappedValue = SharpAIBailingHybridMultiLinear(
            inputDims: kvLora, outputDims: vDim, numHeads: heads)
        _dense.wrappedValue = Linear(heads * vDim, hidden, bias: c.useQKVBias)

        // rope_scaling is not decoded, so the RoPE here is the default one (see file header).
        rope = initializeRope(
            dims: ropeDim, base: c.ropeTheta, traditional: c.ropeInterleave,
            scalingConfig: nil, maxPositionEmbeddings: c.maxPositionEmbeddings)

        super.init()
    }

    /// - Parameter mask: boolean `[L, S]` causal mask, or nil for single-token steps.
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qFull: MLXArray
        if let qProj {
            qFull = qProj(x)
        } else if let qAProj, let qALayernorm, let qBProj {
            qFull = qBProj(qALayernorm(qAProj(x)))
        } else {
            fatalError("bailing_hybrid MLA has no query projection")
        }
        let q = qFull.reshaped(B, L, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(q, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qParts[0]
        var qPe = qParts[1]

        let compressed = kvAProjWithMQA(x)
        let kvParts = split(compressed, indices: [kvLoraRank], axis: -1)
        let kvLatent = kvALayernorm(kvParts[0]).expandedDimensions(axis: 1)  // [B, 1, L, kv]
        var kPe = kvParts[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let ropeOffset = cache?.ropeOffset
        qPe = applyRotaryPosition(rope, to: qPe, offset: ropeOffset)
        kPe = applyRotaryPosition(rope, to: kPe, offset: ropeOffset)

        var keys = kvLatent
        var peKeys = kPe
        if let cache {
            (keys, peKeys) = cache.update(keys: kvLatent, values: kPe)
        }

        var peScores = (qPe * scale).matmul(peKeys.swappedAxes(-1, -2))
        if let mask {
            peScores = MLX.where(
                mask, peScores, MLXArray(-Float.greatestFiniteMagnitude).asType(peScores.dtype))
        }

        let attnQueries: MLXArray
        let attnKeys: MLXArray
        let attnValues: MLXArray
        if L == 1 {
            attnQueries = embedQ(qNope)
            attnKeys = keys
            attnValues = keys
        } else {
            attnQueries = qNope
            attnKeys = embedQ(keys, transpose: false)
            attnValues = unembedOut(keys)
        }

        var output = attentionWithCacheUpdate(
            queries: attnQueries, keys: attnKeys, values: attnValues, cache: nil,
            scale: scale, mask: .array(peScores))
        if L == 1 {
            output = unembedOut(output)
        }
        output = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return dense(output)
    }
}

// MARK: - Other layers (lightning attention)

final class SharpAIBailingHybridLightningAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer
    let _slope: MLXArray

    @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "g_norm") var gNorm: SharpAIBailingHybridGroupRMSNorm
    @ModuleInfo(key: "query_layernorm") var queryLayernorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyLayernorm: RMSNorm?

    init(_ c: SharpAIBailingHybridConfiguration, layerIdx: Int) {
        let heads = c.attentionHeads
        let dim = c.headDim ?? c.hiddenSize / c.attentionHeads

        self.heads = heads
        headDim = dim
        scale = pow(Float(dim), -0.5)

        _queryKeyValue.wrappedValue = Linear(
            c.hiddenSize, 3 * heads * dim, bias: c.useQKVBias)
        _dense.wrappedValue = Linear(heads * dim, c.hiddenSize, bias: c.useBias)
        _gProj.wrappedValue = Linear(c.hiddenSize, heads * dim, bias: false)
        _gNorm.wrappedValue = SharpAIBailingHybridGroupRMSNorm(
            dimensions: heads * dim, eps: c.rmsNormEps, groups: c.groupNormSize ?? 1)

        if c.useQKNorm {
            _queryLayernorm.wrappedValue = RMSNorm(dimensions: dim, eps: c.rmsNormEps)
            _keyLayernorm.wrappedValue = RMSNorm(dimensions: dim, eps: c.rmsNormEps)
        } else {
            _queryLayernorm.wrappedValue = nil
            _keyLayernorm.wrappedValue = nil
        }

        rope = initializeRope(
            dims: Int(Float(dim) * c.partialRotaryFactor), base: c.ropeTheta,
            traditional: false, scalingConfig: nil,
            maxPositionEmbeddings: c.maxPositionEmbeddings)
        _slope = SharpAIBailingHybridLightningAttention.decaySlopes(
            heads: heads, layerIdx: layerIdx, numLayers: c.hiddenLayers)

        super.init()
    }

    /// `_get_slopes` in bailing_hybrid.py: ALiBi-style per-head slopes, scaled by layer depth.
    static func decaySlopes(heads n: Int, layerIdx: Int, numLayers: Int) -> MLXArray {
        func powerOf2Slopes(_ count: Int) -> [Double] {
            let r = pow(2.0, -(log2(Double(count)) - 3.0))
            return (0 ..< count).map { pow(2.0, -r * Double($0 + 1)) }
        }

        let lg = log2(Double(n))
        var slopes: [Double]
        if lg == lg.rounded() {
            slopes = powerOf2Slopes(n)
        } else {
            let p = 1 << Int(lg.rounded(.down))
            let doubled = powerOf2Slopes(2 * p)
            let extra = stride(from: 0, to: doubled.count, by: 2).map { doubled[$0] }
            slopes = powerOf2Slopes(p) + Array(extra.prefix(n - p))
        }

        let denom = max(1, numLayers - 1)
        let layerFactor = Float(1 - Double(layerIdx) / Double(denom) + 1e-5)
        return MLXArray(slopes.map { -Float($0) * layerFactor })
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?, offset: Int) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = queryKeyValue(x).reshaped(B, L, 3 * heads, headDim)
        let parts = split(qkv, indices: [heads, 2 * heads], axis: 2)
        var queries = parts[0].transposed(0, 2, 1, 3)
        var keys = parts[1].transposed(0, 2, 1, 3)
        let values = parts[2].transposed(0, 2, 1, 3)

        if let queryLayernorm, let keyLayernorm {
            queries = queryLayernorm(queries)
            keys = keyLayernorm(keys)
        }

        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(offset))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(offset))

        let (output, newState) = SharpAIBailingHybridRecurrence.recurrentGLA(
            q: queries, k: keys, v: values, g: _slope, scale: scale,
            state: cache?[0], mask: nil)
        cache?[0] = newState

        let flat = output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return dense(gNorm(flat) * sigmoid(gProj(x)))
    }
}

// MARK: - Router and MoE

final class SharpAIBailingHybridGate: Module {
    let topK: Int
    let nGroup: Int
    let topkGroup: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoreFunction: String

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ c: SharpAIBailingHybridConfiguration) {
        topK = c.numExpertsPerTok ?? 1
        nGroup = c.nGroup
        topkGroup = c.topkGroup
        routedScalingFactor = c.routedScalingFactor
        normTopkProb = c.normTopkProb
        scoreFunction = c.scoreFunction

        let experts = c.numExperts ?? 1
        _gateProj.wrappedValue = Linear(c.hiddenSize, experts, bias: false)
        // No expert_bias parameter exists when moe_router_enable_expert_bias is off.
        _expertBias.wrappedValue =
            c.moeRouterEnableExpertBias ? MLXArray.zeros([experts], dtype: .float32) : nil
        super.init()
    }

    /// Returns the selected expert indices and their routing weights, both `[B, L, topK]`.
    func route(_ x: MLXArray) -> (inds: MLXArray, weights: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let logits = gateProj(x)
        let inDType = logits.dtype
        let gates = logits.asType(.float32)
        let origScores =
            scoreFunction == "sigmoid" ? sigmoid(gates) : softmax(gates, axis: -1)

        // The bias steers selection only. The weights come from the unbiased scores.
        var scores = expertBias.map { origScores + $0 } ?? origScores

        let nDrop = nGroup - topkGroup
        if nDrop > 0 {
            let grouped = scores.reshaped(B, L, nGroup, -1)
            let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
            let groupIdx = argPartition(groupScores, kth: nDrop - 1, axis: -2)[
                0..., 0..., ..<nDrop, 0...]
            // -inf, not 0, because the bias can make a kept score negative.
            scores = putAlong(grouped, groupIdx, values: MLXArray(-Float.infinity), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(origScores, inds, axis: -1)
        if topK > 1, normTopkProb {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor
        return (inds, weights.asType(inDType))
    }
}

final class SharpAIBailingHybridSparseMoeBlock: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: SharpAIBailingHybridGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIBailingHybridMLP?

    init(_ c: SharpAIBailingHybridConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: c.hiddenSize,
            hiddenDims: c.moeIntermediateSize ?? c.intermediateSize,
            numExperts: c.numExperts ?? 1,
            bias: c.useBias)
        _gate.wrappedValue = SharpAIBailingHybridGate(c)

        if (c.numSharedExperts ?? 0) > 0 {
            let sharedDim =
                (c.moeSharedExpertIntermediateSize ?? c.moeIntermediateSize ?? c.intermediateSize)
                * (c.numSharedExperts ?? 0)
            _sharedExperts.wrappedValue = SharpAIBailingHybridMLP(
                hiddenSize: c.hiddenSize, intermediateSize: sharedDim, bias: c.useBias)
        } else {
            _sharedExperts.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate.route(x)
        var out = switchMLP(x, inds)
        out = (out * weights[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            out = out + sharedExperts(x)
        }
        return out
    }
}

// MARK: - Decoder layer

final class SharpAIBailingHybridDecoderLayer: Module {
    let isGlobal: Bool

    @ModuleInfo(key: "attention") var attention: Module
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ c: SharpAIBailingHybridConfiguration, layerIdx: Int) {
        let global = c.isGlobalLayer(layerIdx)
        isGlobal = global

        if global {
            _attention.wrappedValue = SharpAIBailingHybridMultiLatentAttention(c)
        } else {
            _attention.wrappedValue = SharpAIBailingHybridLightningAttention(c, layerIdx: layerIdx)
        }

        if c.isMoELayer(layerIdx) {
            _mlp.wrappedValue = SharpAIBailingHybridSparseMoeBlock(c)
        } else {
            _mlp.wrappedValue = SharpAIBailingHybridMLP(
                hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize, bias: c.useBias)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?, offset: Int
    ) -> MLXArray {
        let normed = inputLayerNorm(x)
        let r: MLXArray
        if let mla = attention as? SharpAIBailingHybridMultiLatentAttention {
            r = mla(normed, mask: mask, cache: cache)
        } else if let lightning = attention as? SharpAIBailingHybridLightningAttention {
            r = lightning(normed, cache: cache as? ArraysCache, offset: offset)
        } else {
            fatalError("bailing_hybrid layer has no attention block")
        }
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}
