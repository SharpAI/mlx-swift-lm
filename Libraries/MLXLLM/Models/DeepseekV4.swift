// Copyright © 2025 Apple Inc.

// Port of DeepSeek-V4 inference code
// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct DeepseekV4Configuration: Codable, Sendable {
    // Core architecture
    var vocabSize: Int
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var headDim: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var rmsNormEps: Float

    // Output projection grouping
    var oGroups: Int
    var oLoraRank: Int

    // Attention / compression (per layer)
    var slidingWindow: Int
    var compressRatios: [Int]
    var compressRopeTheta: Float

    // MoE
    var nRoutedExperts: Int
    var nSharedExperts: Int
    var numExpertsPerTok: Int
    var scoringFunc: String
    var routedScalingFactor: Float
    var swiguLimit: Float
    var numHashLayers: Int
    var numNextnPredictLayers: Int
    var normTopkProb: Bool

    // Hyper-Connections (mHC)
    var hcMult: Int
    var hcSinkhornIters: Int
    var hcEps: Float

    // RoPE
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int

    // Nope head dim (derived)
    var nopeHeadDim: Int { headDim - qkRopeHeadDim }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case oGroups = "o_groups"
        case oLoraRank = "o_lora_rank"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case compressRopeTheta = "compress_rope_theta"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case scoringFunc = "scoring_func"
        case routedScalingFactor = "routed_scaling_factor"
        case swiguLimit = "swiglu_limit"
        case numHashLayers = "num_hash_layers"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case normTopkProb = "norm_topk_prob"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
    }
}

// MARK: - Helper Functions

/// sqrtsoftplus activation: sqrt(softplus(x)) = sqrt(log(1 + e^x))
/// Uses numerically stable form to avoid exp overflow for large positive x.
private func sqrtSoftplus(_ x: MLXArray) -> MLXArray {
    let sp = MLX.maximum(x, MLXArray(0)) + MLX.log1p(MLX.exp(-MLX.abs(x)))
    return MLX.sqrt(sp)
}

/// Apply per-head RMS normalization (without learnable scale)
private func headRmsNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
}

// MARK: - Sinkhorn-based Hyper-Connection helpers

/// Split mixes into (pre, post, comb) with Sinkhorn normalization.
/// mixes: [B, S, mix_hc] where mix_hc = (2+hc)*hc
/// Returns pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc]
private func hcSplitSinkhorn(
    _ mixes: MLXArray,
    hcScale: MLXArray,    // [3]
    hcBase: MLXArray,     // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let hc = hcMult
    let B = mixes.dim(0), S = mixes.dim(1)

    // Split mixes into 3 parts
    let preMix = mixes[.ellipsis, ..<hc]            // [B, S, hc]
    let postMix = mixes[.ellipsis, hc ..< 2 * hc]   // [B, S, hc]
    let combMix = mixes[.ellipsis, (2 * hc)...]      // [B, S, hc*hc]

    // Per-part scale, per-element base
    let preBase = hcBase[..<hc]
    let postBase = hcBase[hc ..< 2 * hc]
    let combBase = hcBase[(2 * hc)...]

    // Apply scale, add bias, then sigmoid + eps
    var pre = sigmoid(preMix * hcScale[0] + preBase) + eps   // [B, S, hc]
    let post = sigmoid(postMix * hcScale[1] + postBase) + eps  // [B, S, hc]
    var comb = (sigmoid(combMix * hcScale[2] + combBase) + eps)
        .reshaped(B, S, hc, hc)                               // [B, S, hc, hc]

    // Normalize pre so it sums to 1 across hc copies
    pre = pre / pre.sum(axis: -1, keepDims: true)

    // Sinkhorn normalization for comb (alternating column/row normalize)
    for _ in 0 ..< sinkhornIters {
        comb = comb / comb.sum(axis: -2, keepDims: true)  // column normalize
        comb = comb / comb.sum(axis: -1, keepDims: true)  // row normalize
    }

    return (pre, post, comb)
}

/// Hyper-Connection pre-step: reduce [B,S,hc,D] → [B,S,D] with Sinkhorn weights.
/// Returns (reduced_x, post_weights, comb_matrix).
private func hcPre(
    x: MLXArray,           // [B, S, hc, D]
    hcFn: MLXArray,        // [mix_hc, hc*D]
    hcScale: MLXArray,     // [3]
    hcBase: MLXArray,      // [mix_hc]
    hcMult: Int,
    sinkhornIters: Int,
    eps: Float
) -> (MLXArray, MLXArray, MLXArray) {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    // Flatten: [B, S, hc*D]
    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)

    // RMS-style normalization scale
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)

    // Linear projection: [B, S, mix_hc]
    let mixes = matmul(xFlat, hcFn.T) * normScale

    let (pre, post, comb) = hcSplitSinkhorn(
        mixes, hcScale: hcScale, hcBase: hcBase,
        hcMult: hcMult, sinkhornIters: sinkhornIters, eps: eps)

    // Weighted sum of hc copies: [B, S, D]
    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)

    return (y, post, comb)
}

/// Hyper-Connection post-step: expand sublayer output back to [B,S,hc,D].
/// y[b,s,j,:] = post[b,s,j]*x[b,s,:] + sum_i(comb[b,s,i,j]*residual[b,s,i,:])
private func hcPost(
    x: MLXArray,        // [B, S, D] - sublayer output
    residual: MLXArray, // [B, S, hc, D] - input to this block
    post: MLXArray,     // [B, S, hc]
    comb: MLXArray      // [B, S, hc, hc]
) -> MLXArray {
    // term1: post[b,s,j] * x[b,s,:] → broadcast to [B,S,hc,D]
    let term1 = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2)

    // term2: sum_i(comb[b,s,i,j] * residual[b,s,i,:])
    // comb.unsqueeze(-1): [B,S,hc_i,hc_j,1]
    // residual.unsqueeze(-2): [B,S,hc_i,1,D]
    // product: [B,S,hc_i,hc_j,D] → sum over dim 2 → [B,S,hc_j,D]
    let combExp = comb.expandedDimensions(axis: -1)         // [B,S,hc,hc,1]
    let residualExp = residual.expandedDimensions(axis: -2) // [B,S,hc,1,D]
    let term2 = (combExp * residualExp).sum(axis: 2)        // [B,S,hc,D]

    return (term1 + term2).asType(x.dtype)
}

// MARK: - HCParams Module

/// Lightweight Module to hold the three Hyper-Connection tensors loaded from checkpoint.
/// Key names (fn, base, scale) match the `hc_attn.*` / `hc_ffn.*` / `hc_head.*` paths.
class HCParams: Module {
    var fn: MLXArray
    var base: MLXArray
    var scale: MLXArray

    init(fn: MLXArray, base: MLXArray, scale: MLXArray) {
        self.fn = fn
        self.base = base
        self.scale = scale
    }
}

/// Final HC head: reduce [B,S,hc,D] → [B,S,D] for lm_head.
/// No Sinkhorn – just sigmoid + eps weighted sum.
private func hcHead(
    x: MLXArray,        // [B, S, hc, D]
    hcFn: MLXArray,     // [hc, hc*D]
    hcScale: MLXArray,  // [1]
    hcBase: MLXArray,   // [hc]
    eps: Float
) -> MLXArray {
    let dtype = x.dtype
    let B = x.dim(0), S = x.dim(1), hc = x.dim(2), D = x.dim(3)

    let xFlat = x.reshaped(B, S, hc * D).asType(.float32)
    let normScale = rsqrt(xFlat.square().mean(axis: -1, keepDims: true) + eps)
    let mixes = matmul(xFlat, hcFn.T) * normScale           // [B, S, hc]
    let pre = sigmoid(mixes * hcScale + hcBase) + eps        // [B, S, hc]

    // Weighted sum: [B, S, D]
    let y = (pre.expandedDimensions(axis: -1).asType(dtype) * x).sum(axis: -2)
    return y.asType(dtype)
}

// MARK: - Attention

/// Attention with cache update that optionally applies per-head sink bias.
/// Mirrors `attentionWithCacheUpdateAndSinks` from MiMoV2Flash but uses the public API.
private func deepseekAttentionWithSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    sinks: MLXArray?
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask, sinks: sinks)
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (qk, qv) = quantizedKVCache.updateQuantized(keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: qk, quantizedValues: qv,
            scale: scale, mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode)
    }
    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries, keys: cachedKeys, values: cachedValues,
        scale: scale, mask: mask, sinks: sinks)
}

class DeepseekV4Attention: Module {
    let config: DeepseekV4Configuration
    let numHeads: Int
    let headDim: Int
    let nopeHeadDim: Int
    let ropeHeadDim: Int
    let oGroups: Int
    let oLoraRank: Int
    let nHeadsPerGroup: Int
    let scale: Float
    let eps: Float

    let rope: RoPELayer

    // Q low-rank projections
    @ModuleInfo(key: "wq_a") var wqA: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "wq_b") var wqB: Linear

    // Unified KV projection (K and V share the same projection)
    @ModuleInfo(key: "wkv") var wkv: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    // Grouped output projection
    @ModuleInfo(key: "wo_a") var woA: Linear
    @ModuleInfo(key: "wo_b") var woB: Linear

    // Attention sink bias (per head, no .weight suffix)
    // Stored via update(parameters:) using the key "attn_sink"
    var attn_sink: MLXArray

    init(config: DeepseekV4Configuration) {
        self.config = config
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.nopeHeadDim = config.nopeHeadDim
        self.ropeHeadDim = config.qkRopeHeadDim
        self.oGroups = config.oGroups
        self.oLoraRank = config.oLoraRank
        self.nHeadsPerGroup = config.numAttentionHeads / config.oGroups
        self.scale = pow(Float(config.headDim), -0.5)
        self.eps = config.rmsNormEps

        // Q projections
        self._wqA.wrappedValue = Linear(config.hiddenSize, config.qLoraRank, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: config.rmsNormEps)
        self._wqB.wrappedValue = Linear(config.qLoraRank, config.numAttentionHeads * config.headDim, bias: false)

        // Unified KV: single head, headDim dimensional
        self._wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)

        // Grouped output projection
        // wo_a: Linear(nHeadsPerGroup * headDim, oGroups * oLoraRank) per group → stored as [oGroups*oLoraRank, nHeadsPerGroup*headDim]
        self._woA.wrappedValue = Linear(nHeadsPerGroup * config.headDim, config.oGroups * config.oLoraRank, bias: false)
        self._woB.wrappedValue = Linear(config.oGroups * config.oLoraRank, config.hiddenSize, bias: false)

        // Attention sink: per-head bias [numAttentionHeads], applied to attention logits before softmax.
        // Shape matches numAttentionHeads (== qkRopeHeadDim in this architecture).
        self.attn_sink = zeros([config.numAttentionHeads])

        // RoPE using compress_rope_theta (used for most layers with compress_ratio != 0)
        // We use a single rope config as a simplification
        let ropeBase = config.compressRopeTheta
        self.rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: ropeBase,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    /// Grouped output projection matching the reference Python implementation.
    /// For QuantizedLinear wo_a: slices weight rows per group, calls quantizedMM.
    /// For plain Linear wo_a: uses batched matmul after weight reshape.
    /// Input:  [B, L, n_heads, head_dim]
    /// Output: [B, L, oGroups * oLoraRank]
    private func groupedOutputProjection(_ out: MLXArray) -> MLXArray {
        let B = out.dim(0), L = out.dim(1)
        let groupFeat = numHeads * headDim / oGroups  // = nHeadsPerGroup * headDim

        // Flatten to [B, L, n_heads * head_dim] for easy group slicing
        let outFlat = out.reshaped(B, L, numHeads * headDim)

        if let qLinear = woA as? QuantizedLinear {
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd   = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd   = (g + 1) * oLoraRank

                // Per-group input: [B, L, groupFeat]
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]
                // Slice weight rows for this group
                let wRows = qLinear.weight[rStart ..< rEnd]
                let sRows = qLinear.scales[rStart ..< rEnd]
                let bRows = qLinear.biases.map { $0[rStart ..< rEnd] }

                // quantizedMM: [B, L, groupFeat] @ dequant(wRows)^T → [B, L, oLoraRank]
                let y = quantizedMM(
                    groupInput,
                    wRows,
                    scales: sRows,
                    biases: bRows,
                    transpose: true,
                    groupSize: qLinear.groupSize,
                    bits: qLinear.bits,
                    mode: qLinear.mode
                )
                pieces.append(y)
            }
            return concatenated(pieces, axis: -1)  // [B, L, oGroups * oLoraRank]
        } else {
            // Non-quantized fallback: per-group matmul (same structure as quantized path).
            // A single batched matmul would broadcast batch dims [B,L] against [oGroups],
            // which fails when L != oGroups, so we loop instead.
            var pieces: [MLXArray] = []
            for g in 0 ..< oGroups {
                let gStart = g * groupFeat
                let gEnd   = (g + 1) * groupFeat
                let rStart = g * oLoraRank
                let rEnd   = (g + 1) * oLoraRank
                let groupInput = outFlat[0..., 0..., gStart ..< gEnd]  // [B, L, groupFeat]
                let wa_g = woA.weight[rStart ..< rEnd]                 // [oLoraRank, groupFeat]
                pieces.append(matmul(groupInput, wa_g.T))              // [B, L, oLoraRank]
            }
            return concatenated(pieces, axis: -1)
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        // --- Query ---
        // Low-rank Q: wq_a → q_norm → wq_b
        var q = wqB(qNorm(wqA(x)))                       // [B, L, n_heads * head_dim]
        q = q.reshaped(B, L, numHeads, headDim)
            .transposed(0, 2, 1, 3)                       // [B, n_heads, L, head_dim]
        // Per-head RMS normalization (no learnable scale)
        q = headRmsNorm(q, eps: eps)

        // Split Q into nope and rope parts
        let qNope = q[.ellipsis, ..<nopeHeadDim]          // [B, n_heads, L, nope_head_dim]
        var qRope = q[.ellipsis, nopeHeadDim...]           // [B, n_heads, L, rope_head_dim]
        qRope = applyRotaryPosition(rope, to: qRope, cache: cache)
        let queries = concatenated([qNope, qRope], axis: -1) // [B, n_heads, L, head_dim]

        // --- KV: k = v (reference: k = v = concat([k_nope, k_pe_roped])) ---
        let kv = kvNorm(wkv(x))                           // [B, L, head_dim]
        let kvNope = kv[.ellipsis, ..<nopeHeadDim]
            .reshaped(B, L, 1, nopeHeadDim)
            .transposed(0, 2, 1, 3)                       // [B, 1, L, nope_head_dim]
        var kvRope = kv[.ellipsis, nopeHeadDim...]
            .reshaped(B, L, 1, ropeHeadDim)
            .transposed(0, 2, 1, 3)                       // [B, 1, L, rope_head_dim]
        kvRope = applyRotaryPosition(rope, to: kvRope, cache: cache)
        let kFull = concatenated([kvNope, kvRope], axis: -1) // [B, 1, L, head_dim]
        // In reference k = v = kFull: both K and V have rope applied to their rope dims.
        // attentionWithCacheUpdate handles the KV cache update internally.

        // --- Attention ---
        // Pass kFull as both keys and values; cache update happens inside.
        // Apply attn_sink (per-head bias) to attention logits when non-zero.
        // Cast to queries.dtype (bfloat16) — attn_sink may be loaded as float32 from the
        // checkpoint, but MLXFast.scaledDotProductAttention requires sinks to promote to
        // the output dtype (bfloat16); float32 does not satisfy this constraint.
        let sinksToUse: MLXArray? = attn_sink.sum().item(Float.self) != 0
            ? attn_sink.asType(queries.dtype)
            : nil
        let output = deepseekAttentionWithSinks(
            queries: queries,
            keys: kFull,
            values: kFull,
            cache: cache,
            scale: scale,
            mask: mask,
            sinks: sinksToUse
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, numHeads, headDim)               // [B, L, n_heads, head_dim]

        // --- Grouped output projection ---
        let oLora = groupedOutputProjection(output)        // [B, L, oGroups * oLoraRank]
        return woB(oLora)
    }
}

// MARK: - MoE Components

/// Single FFN expert: SwiGLU with optional activation clamping.
class DeepseekV4Expert: Module, UnaryLayer {
    let swiguLimit: Float

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int, swiguLimit: Float) {
        self.swiguLimit = swiguLimit
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gate = gateProj(x)
        var up = upProj(x)
        if swiguLimit > 0 {
            gate = clip(gate, min: -swiguLimit, max: swiguLimit)
            up = clip(up, min: -swiguLimit, max: swiguLimit)
        }
        return downProj(silu(gate) * up)
    }
}

/// MoE routing gate with sqrtsoftplus (score-based routing only; hash routing not implemented).
class DeepseekV4Gate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool
    let scoringFunc: String

    var weight: MLXArray              // [n_routed_experts, hidden_size]
    var e_score_correction_bias: MLXArray  // [n_routed_experts]

    init(config: DeepseekV4Configuration) {
        self.topK = config.numExpertsPerTok
        self.nRoutedExperts = config.nRoutedExperts
        self.routedScalingFactor = config.routedScalingFactor
        self.normTopkProb = config.normTopkProb
        self.scoringFunc = config.scoringFunc
        self.weight = zeros([config.nRoutedExperts, config.hiddenSize])
        self.e_score_correction_bias = zeros([config.nRoutedExperts])
    }

    /// Allow hash-routing layers (0..numHashLayers-1) to load without e_score_correction_bias.
    /// Those layers use tid2eid hash routing in the original code; we keep the zero default.
    override func updateMissing(
        parameter: String,
        verify: VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws {
        if parameter == "e_score_correction_bias" {
            return  // keep zero-initialized default
        }
        try super.updateMissing(
            parameter: parameter, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // Compute expert scores
        let logits = x.matmul(weight.T)       // [B, S, n_experts]
        var scores: MLXArray
        switch scoringFunc {
        case "softmax":
            scores = softmax(logits, axis: -1)
        case "sigmoid":
            scores = sigmoid(logits)
        default:
            // sqrtsoftplus: sqrt(softplus(x)) = sqrt(log(1 + e^x))
            scores = sqrtSoftplus(logits)
        }

        // Bias-shifted scores for top-k selection (bias not applied to routing weights)
        let scoresForChoice = scores + e_score_correction_bias

        // Top-k selection
        let inds = argPartition(-scoresForChoice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        // Gather weights using original (non-biased) scores
        var selectedScores = takeAlong(scores, inds, axis: -1)

        if topK > 1 && normTopkProb {
            let denominator = selectedScores.sum(axis: -1, keepDims: true) + 1e-20
            selectedScores = selectedScores / denominator
        }
        selectedScores = selectedScores * routedScalingFactor

        return (inds, selectedScores)
    }
}

/// Mixture-of-Experts layer with shared expert.
class DeepseekV4MoE: Module, UnaryLayer {
    let numExpertsPerTok: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    var gate: DeepseekV4Gate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV4Expert

    init(config: DeepseekV4Configuration) {
        self.numExpertsPerTok = config.numExpertsPerTok

        // Routed experts (stacked via SwitchGLU, same as V3)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts,
            activation: { x in
                // SwiGLU with limit
                if config.swiguLimit > 0 {
                    let g = clip(x, min: -config.swiguLimit, max: config.swiguLimit)
                    return silu(g)
                }
                return silu(x)
            }
        )
        self.gate = DeepseekV4Gate(config: config)

        // Shared expert (1 expert, same intermediate size)
        self._sharedExperts.wrappedValue = DeepseekV4Expert(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.moeIntermediateSize,
            swiguLimit: config.swiguLimit
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        // Add shared expert output
        y = y + sharedExperts(x)
        return y
    }
}

// MARK: - Decoder Block (with mHC Hyper-Connections)

class DeepseekV4Block: Module {
    let config: DeepseekV4Configuration

    // Key "attn" matches checkpoint path `layers.{l}.attn.*`
    @ModuleInfo(key: "attn") var selfAttn: DeepseekV4Attention
    // Plain var: property name "ffn" matches checkpoint path `layers.{l}.ffn.*`
    var ffn: DeepseekV4MoE
    // Key names match checkpoint: `attn_norm`, `ffn_norm`
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

    // Hyper-Connection parameter bundles.
    // Underscore names match checkpoint paths: `hc_attn.fn/base/scale`, `hc_ffn.fn/base/scale`
    var hc_attn: HCParams
    var hc_ffn: HCParams

    init(config: DeepseekV4Configuration) {
        self.config = config

        self._selfAttn.wrappedValue = DeepseekV4Attention(config: config)
        self.ffn = DeepseekV4MoE(config: config)

        self._attnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._ffnNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // Initialize HC parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        let mixHc = (2 + hc) * hc
        let hcDim = hc * config.hiddenSize
        self.hc_attn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
        self.hc_ffn = HCParams(
            fn: zeros([mixHc, hcDim]),
            base: zeros([mixHc]),
            scale: ones([3]))
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        // x: [B, S, hc, D]
        let residualAttn = x

        // HC pre for attention: [B,S,hc,D] → [B,S,D]
        let (xAttn, postAttn, combAttn) = hcPre(
            x: residualAttn,
            hcFn: hc_attn.fn,
            hcScale: hc_attn.scale,
            hcBase: hc_attn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // Attention sublayer: [B,S,D] → [B,S,D]
        let attnOut = selfAttn(attnNorm(xAttn), mask: mask, cache: cache)

        // HC post for attention: [B,S,D] → [B,S,hc,D]
        let residualFfn = hcPost(x: attnOut, residual: residualAttn, post: postAttn, comb: combAttn)

        // HC pre for FFN: [B,S,hc,D] → [B,S,D]
        let (xFfn, postFfn, combFfn) = hcPre(
            x: residualFfn,
            hcFn: hc_ffn.fn,
            hcScale: hc_ffn.scale,
            hcBase: hc_ffn.base,
            hcMult: config.hcMult,
            sinkhornIters: config.hcSinkhornIters,
            eps: config.hcEps
        )

        // FFN sublayer: [B,S,D] → [B,S,D]
        let ffnOut = ffn(ffnNorm(xFfn))

        // HC post for FFN: [B,S,D] → [B,S,hc,D]
        return hcPost(x: ffnOut, residual: residualFfn, post: postFfn, comb: combFfn)
    }
}

// MARK: - Inner Model

public class DeepseekV4ModelInner: Module, LayerPartitionable, StreamableMoE {
    var config: DeepseekV4Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    // HC head parameter bundle for final reduction [B,S,hc,D] → [B,S,D]
    // Underscore name matches checkpoint path `model.hc_head.fn/base/scale`
    var hc_head: HCParams

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { layers.count }

    init(config: DeepseekV4Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        // Exclude MTP (multi-token prediction) layers from the main transformer stack
        let mainLayerCount = config.numHiddenLayers - config.numNextnPredictLayers
        self.layers = (0 ..< mainLayerCount).map {
            _ in DeepseekV4Block(config: config)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        // HC head parameters (will be overwritten by weight loading)
        let hc = config.hcMult
        self.hc_head = HCParams(
            fn: zeros([hc, hc * config.hiddenSize]),
            base: zeros([hc]),
            scale: ones([1]))
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        // x: [B, S] token IDs
        let B = x.dim(0), S = x.dim(1)
        let hc = config.hcMult

        // Embed tokens: [B, S, D]
        var h = embedTokens(x)

        // Expand to hc copies: [B, S, hc, D]
        // Repeat along new hc dimension
        h = h.expandedDimensions(axis: 2)                  // [B, S, 1, D]
        h = repeated(h, count: hc, axis: 2)                // [B, S, hc, D]

        // Create causal attention mask; reshape to 3D so dim(1)==S
        let hForMask = h.reshaped([B, S, hc * config.hiddenSize])  // [B, S, hc*D]
        let attentionMask = createAttentionMask(h: hForMask, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(
                index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(h, mask: attentionMask, cache: cache?[i])
            }
        }

        // HC head: [B, S, hc, D] → [B, S, D]
        h = hcHead(
            x: h, hcFn: hc_head.fn, hcScale: hc_head.scale,
            hcBase: hc_head.base, eps: config.hcEps)

        return norm(h)
    }
}

// MARK: - Top-level Model

public class DeepseekV4Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    /// One KV head per layer (unified KV, single head)
    public var kvHeads: [Int]

    var args: DeepseekV4Configuration
    public var model: DeepseekV4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV4Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: 1, count: args.numHiddenLayers - args.numNextnPredictLayers)
        self.model = DeepseekV4ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        // 1. Dequantize FP8 weights (weight_scale_inv pattern, same as V3)
        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs
            var padded = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    newWeights[weightKey] = dequant(weight: weight, scaleInv: value)
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        // 2. Stack per-expert weights into SwitchGLU format (for non-pre-stacked checkpoints)
        // MLX quantized checkpoints already have stacked weights; this is a no-op for them.
        let mainLayerCount = args.numHiddenLayers - args.numNextnPredictLayers
        for l in 0 ..< mainLayerCount {
            let prefix = "model.layers.\(l)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).ffn.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let stacked = (0 ..< args.nRoutedExperts).map {
                            // Prefer dequantized value from newWeights (FP8 dequant), fall back to original
                            newWeights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]
                                ?? weights["\(prefix).ffn.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).ffn.switch_mlp.\(projName).\(key)"] = MLX.stacked(stacked)
                        for j in 0 ..< args.nRoutedExperts {
                            newWeights.removeValue(forKey: "\(prefix).ffn.experts.\(j).\(projName).\(key)")
                        }
                    }
                }
            }
        }

        // 3. Filter out MTP (multi-token prediction) layers and rotary_emb keys
        // Also drop compressor/indexer sub-module keys (not yet implemented)
        let numMainLayers = args.numHiddenLayers - args.numNextnPredictLayers
        return newWeights.filter { key, _ in
            // Drop MTP layer weights (layers at index >= numMainLayers)
            if key.starts(with: "model.layers.") && !MTPConfig.retainMTPWeights {
                let parts = key.split(separator: ".")
                if parts.count >= 3, let layerIdx = Int(parts[2]) {
                    if layerIdx >= numMainLayers {
                        return false
                    }
                }
            }
            // Drop rotary embedding precomputed frequencies
            if key.contains("rotary_emb.inv_freq") { return false }
            // Drop compressor/indexer sub-module weights — these implement long-range
            // compressed attention and are not yet implemented in this Swift port.
            // Affected layers are those with compress_ratio != 0 (layers 2+).
            // TODO: implement DeepseekV4Compressor and DeepseekV4Indexer modules.
            if key.contains(".attn.compressor.") || key.contains(".attn.indexer.") {
                return false
            }
            // Note: .attn.attn_sink is a valid model parameter — do NOT filter it.
            // Drop gate.tid2eid — hash-layer token-to-expert lookup table (not yet implemented).
            // Hash layers (0..numHashLayers-1) use deterministic routing; we fall back to
            // the learned gate.weight for these layers instead.
            if key.contains(".ffn.gate.tid2eid") { return false }
            return true
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - MTPLanguageModel Conformance for DeepseekV4Model

/// DeepSeek V4 uses a different MTP scheme: the MTP layers are the last
/// `numNextnPredictLayers` standard transformer blocks (`model.layers[numMainLayers...]`).
/// They share the same architecture as the main blocks but operate on the final hidden state.
/// The main `lm_head` is reused for all MTP depth projections.
extension DeepseekV4Model: MTPLanguageModel {
    public func callMTP(_ inputs: MLXArray, cache: [KVCache]?) -> [MLXArray] {
        let numMain = args.numHiddenLayers - args.numNextnPredictLayers
        guard MTPConfig.retainMTPWeights, args.numNextnPredictLayers > 0 else {
            return [callAsFunction(inputs, cache: cache)]
        }

        // Run the main model body (excludes MTP layers \u2014 DeepseekV4ModelInner only
        // instantiates `numMain` blocks, so this is the standard forward pass)
        let mainHidden = model(inputs, cache: cache)
        let mainLogits = lmHead(mainHidden)
        var result = [mainLogits]

        // Chain MTP blocks stored in `model.layers` beyond numMain.
        // Note: DeepseekV4ModelInner.layers only contains the first numMain layers.
        // The MTP layer weights land in the weight dict as model.layers.{numMain+i}.*
        // but are not currently allocated as Module objects inside DeepseekV4ModelInner.
        // Phase 2 completion: allocate a separate `mtpLayers: [DeepseekV4Block]` array
        // in DeepseekV4ModelInner when MTPConfig.retainMTPWeights is true.
        // For now, return only the main logits (safe fallback).
        return result
    }
}
