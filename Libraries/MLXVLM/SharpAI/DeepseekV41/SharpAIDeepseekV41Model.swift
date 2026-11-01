// SharpAI fork overlay: deepseek_v41 text model. Not part of upstream mlx-swift-lm.
//
// Port of the text path of ml-explore/mlx-lm mlx_lm/models/deepseek_v41.py (Copyright 2026 Apple Inc.).
// Verified against: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/deepseek_v41.py
// Verified against: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/rope_utils.py
// Verified against: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/activations.py
//
// Ported: sliding-window attention with attention sinks, the compressor and indexer (including
// the candidate-block source layer), the grouped low-rank output projection, interleaved YaRN
// RoPE, the hyper-connection residual streams (mHC), the sqrtsoftplus router with the clamped
// SwiGLU experts and the shared expert, and the per-layer incremental cache.
//
// Not ported (KNOWN GAPs are in the test files): fp8/fp4 activation fake-quant, pipeline and
// tensor sharding, the Metal sinkhorn kernel (the reference CPU loop is used, same arithmetic),
// Engram (the creator rejects it, as the reference does), the MTP and DSpark heads, the vision
// tower, and prompt-cache serialization.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Math helpers

/// RMSNorm without a learned weight (the reference's `mx.fast.rms_norm(x, weight=None)`).
private func deepseekV41RMSNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
    x * rsqrt(mean(x * x, axis: -1, keepDims: true) + eps)
}

/// Inverse frequencies for interleaved RoPE, computed as mlx-lm's YarnRoPE does. With a factor
/// of 1 they reduce to base^(2i / dims).
func deepseekV41RopeFrequencies(
    dims: Int, base: Float, factor: Float, originalMaxPositionEmbeddings: Int,
    betaFast: Float, betaSlow: Float
) -> [Float] {
    let d = Double(dims)
    let b = Double(base)
    func correctionDim(_ rotations: Double) -> Double {
        d * log(Double(originalMaxPositionEmbeddings) / (rotations * 2 * Double.pi))
            / (2 * log(b))
    }
    let low0 = max(floor(correctionDim(Double(betaFast))), 0)
    let high0 = min(ceil(correctionDim(Double(betaSlow))), d - 1)
    let low = low0
    var high = high0
    if low == high {
        high += 0.001
    }
    let half = dims / 2
    var freqs = [Float](repeating: 0, count: half)
    for i in 0 ..< half {
        let extra = pow(b, Double(2 * i) / d)
        let inter = Double(factor) * extra
        let ramp = min(max((Double(i) - low) / (high - low), 0), 1)
        let mask = 1 - ramp
        freqs[i] = Float((inter * extra) / (inter * mask + extra * (1 - mask)))
    }
    return freqs
}

/// Interleaved RoPE on the trailing `ropeDim` features (the reference `_apply_rope`).
/// `x` is [B, S, D] or [B, S, H, D]; the sequence axis is the second one.
func deepseekV41ApplyRope(
    _ x: MLXArray, freqs: [Float], ropeDim: Int, offset: Int, scale: Float = 1,
    inverse: Bool = false
) -> MLXArray {
    if ropeDim == 0 {
        return x
    }
    let width = x.dim(-1)
    let head = x[.ellipsis, 0 ..< (width - ropeDim)]
    let tail = x[.ellipsis, (width - ropeDim)...].asType(.float32)
    let table = MLXArray(inverse ? freqs.map { -$0 } : freqs)
    let rotated: MLXArray
    if x.ndim == 4 {
        rotated = MLXFast.RoPE(
            tail.transposed(0, 2, 1, 3), dimensions: ropeDim, traditional: true, base: nil,
            scale: scale, offset: offset, freqs: table
        ).transposed(0, 2, 1, 3)
    } else {
        rotated = MLXFast.RoPE(
            tail, dimensions: ropeDim, traditional: true, base: nil, scale: scale,
            offset: offset, freqs: table)
    }
    return concatenated([head, rotated.asType(x.dtype)], axis: -1)
}

/// Sinkhorn normalization of the comb matrices: the reference CPU path. The column normalization
/// comes first, then iters - 1 rounds of row and column normalization.
func deepseekV41Sinkhorn(_ comb: MLXArray, iters: Int, eps: Float) -> MLXArray {
    var c = comb / (sum(comb, axis: -2, keepDims: true) + eps)
    if iters > 1 {
        for _ in 0 ..< (iters - 1) {
            c = c / (sum(c, axis: -1, keepDims: true) + eps)
            c = c / (sum(c, axis: -2, keepDims: true) + eps)
        }
    }
    return c
}

/// `_collapse` of the reference: [B,S,hc,D] weighted by [B,S,hc] -> [B,S,D].
func deepseekV41Collapse(_ streams: MLXArray, _ mix: MLXArray) -> MLXArray {
    let out = sum(mix.expandedDimensions(axis: -1) * streams.asType(.float32), axis: 2)
    return out.asType(streams.dtype)
}

/// `_expand` of the reference: mixes the residual streams with `comb` and adds the sublayer
/// output `x` weighted by `post`.
func deepseekV41Expand(
    _ x: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
) -> MLXArray {
    let mixed = sum(
        comb.expandedDimensions(axis: -1) * residual.expandedDimensions(axis: -2), axis: 2)
    let out = post.expandedDimensions(axis: -1) * x.expandedDimensions(axis: -2) + mixed
    return out.asType(residual.dtype)
}

/// The clamped SwiGLU of DeepSeek-V4.1: the gate is clipped from above and the up branch from
/// both sides, then silu(gate) * up.
func deepseekV41ClampedSwiGLU(up: MLXArray, gate: MLXArray, limit: Float) -> MLXArray {
    let dtype = up.dtype
    var g = gate.asType(.float32)
    var u = up.asType(.float32)
    if limit > 0 {
        g = minimum(g, limit)
        u = clip(u, min: -limit, max: limit)
    }
    return (silu(g) * u).asType(dtype)
}

/// Rows of `values` [B, K, D] at `indices` [B, S, k] -> [B, S, k, D]. An index of -1 gathers
/// row 0; the caller masks those entries out.
func deepseekV41GatherRows(_ values: MLXArray, _ indices: MLXArray) -> MLXArray {
    let batch = values.dim(0)
    let length = values.dim(1)
    let dim = values.dim(2)
    let safe = maximum(indices, MLXArray(Int32(0)))
    let base = MLXArray((0 ..< batch).map { Int32($0 * length) }).reshaped(batch, 1, 1)
    let flat = values.reshaped(batch * length, dim)
    let picked = flat[(safe + base).reshaped(-1)]
    return picked.reshaped(indices.shape + [dim])
}

/// Attention over the rows each query picks from each KV array. `q` is [B, S, H, D], each KV
/// array [B, K_i, D] with its indices [B, S, k_i]. An index of -1 picks nothing. The sinks are
/// extra logits that take softmax mass and have no value, as in the reference SDPA.
func deepseekV41SparseAttention(
    _ q: MLXArray, kvs: [MLXArray], indices: [MLXArray], sinks: MLXArray, scale: Float
) -> MLXArray {
    let batch = q.dim(0)
    let length = q.dim(1)
    let heads = q.dim(2)
    let rows = zip(kvs, indices).map { deepseekV41GatherRows($0, $1) }
    // Each query has its own keys: [B, S, K, D] with K = sum of k_i.
    let keys = concatenated(rows, axis: 2).asType(.float32)
    let valid: MLXArray = concatenated(indices, axis: -1) .>= MLXArray(Int32(0))
    let qf = q.asType(.float32)
    var scores = matmul(qf, keys.transposed(0, 1, 3, 2)) * scale  // [B, S, H, K]
    scores = which(valid.expandedDimensions(axis: 2), scores, -Float.infinity)
    let sinkColumn = broadcast(
        sinks.asType(.float32).reshaped(1, 1, heads, 1), to: [batch, length, heads, 1])
    let all = concatenated([scores, sinkColumn], axis: -1)
    let probs = softmax(all, axis: -1)[.ellipsis, 0 ..< keys.dim(2)]
    return matmul(probs, keys).asType(q.dtype)  // [B, S, H, D]
}

/// Window rows each query attends to, -1 for none. The window holds the `previous` rows, then the
/// `length` new ones, so query t is at row previous + t.
func deepseekV41WindowIndices(batch: Int, previous: Int, length: Int, window: Int) -> MLXArray {
    let width = min(window, previous + length)
    var data: [Int32] = []
    data.reserveCapacity(batch * length * width)
    for _ in 0 ..< batch {
        for t in 0 ..< length {
            let end = previous + t
            let start = max(end - window + 1, 0)
            for j in 0 ..< width {
                let index = start + j
                data.append(index > end ? -1 : Int32(index))
            }
        }
    }
    return MLXArray(data, [batch, length, width])
}

private func deepseekV41Append(_ cached: MLXArray?, _ new: MLXArray) -> MLXArray {
    guard let cached else {
        return new
    }
    return concatenated([cached, new], axis: 1)
}

// MARK: - Cache

/// Cache for one decoder layer: the sliding window, the closed compressed groups and their index
/// keys, and the rows of the group that is still open. Its state is not trimmable and not
/// serialized (the reference has the same limits).
public final class DeepseekV41LayerCache: BaseKVCache {
    let windowSize: Int
    var window: MLXArray?
    var compressKV: MLXArray?
    var indexK: MLXArray?
    var pendingKV: MLXArray?
    var pendingGate: MLXArray?

    public init(windowSize: Int) {
        self.windowSize = windowSize
        super.init()
    }

    /// Returns the window with `kv` appended, and keeps the last `windowSize` rows.
    func updateWindow(_ kv: MLXArray) -> MLXArray {
        let full = deepseekV41Append(window, kv)
        let keep = min(windowSize, full.dim(1))
        window = full[0..., (full.dim(1) - keep)...]
        offset += kv.dim(1)
        return full
    }

    public override func innerState() -> [MLXArray] {
        [window, compressKV, indexK, pendingKV, pendingGate].compactMap { $0 }
    }
}

// MARK: - Attention pieces

/// A block-diagonal projection: one [out, in] matrix per output group.
public final class DeepseekV41BlockLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(inputDims: Int, outputDims: Int, groups: Int) {
        self._weight.wrappedValue = MLXArray.zeros(
            [groups, outputDims, inputDims], type: Float.self)
        super.init()
    }

    /// [N, G, in] -> [N, G, out]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        matmul(x.expandedDimensions(axis: -2), weight.transposed(0, 2, 1)).squeezed(axis: -2)
    }
}

public final class DeepseekV41Compressor: Module {
    let ratio: Int
    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm

    init(_ args: SharpAIDeepseekV41TextConfiguration, layerId: Int) {
        ratio = args.compressRatios[layerId]
        self._kvProj.wrappedValue = Linear(args.hiddenSize, args.headDim, bias: false)
        self._gateProj.wrappedValue =
            ratio > 1 ? Linear(args.hiddenSize, args.headDim, bias: false) : nil
        self._kvNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        super.init()
    }

    /// Pools each full group of `ratio` tokens into one latent. Returns the latents (nil when no
    /// group is full) and the index of the first group.
    func callAsFunction(
        _ x: MLXArray, startPos: Int, cache: DeepseekV41LayerCache
    ) -> (MLXArray?, Int) {
        if ratio == 1 {
            return (kvNorm(kvProj(x)), startPos)
        }
        guard let gateProj else {
            fatalError("deepseek_v41 compressor with ratio \(ratio) has no gate_proj")
        }
        let h = x.asType(.float32)
        let pending = cache.pendingKV?.dim(1) ?? 0
        let kv = deepseekV41Append(cache.pendingKV, kvProj(h))
        let gate = deepseekV41Append(cache.pendingGate, gateProj(h))
        let full = kv.dim(1) - kv.dim(1) % ratio
        cache.pendingKV = kv[0..., full...]
        cache.pendingGate = gate[0..., full...]
        // A group takes the position of its first token. The RoPE scale multiplies the offset
        // by the ratio, so the group index is returned.
        let first = (startPos - pending) / ratio
        if full == 0 {
            return (nil, first)
        }
        let batch = kv.dim(0)
        let dim = kv.dim(2)
        let groups = kv[0..., 0 ..< full].reshaped(batch, -1, ratio, dim)
        let gates = gate[0..., 0 ..< full].reshaped(batch, -1, ratio, dim)
        let latent = sum(groups * softmax(gates, axis: 2), axis: 2)
        return (kvNorm(latent.asType(x.dtype)), first)
    }
}

public final class DeepseekV41Indexer: Module {
    let ratio: Int
    let isCandidateSource: Bool
    let candidateTopkBlocks: Int
    let candidateBlockSize: Int
    let numHeads: Int
    let headDim: Int
    let indexTopk: Int
    let ropeDim: Int
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    init(_ args: SharpAIDeepseekV41TextConfiguration, layerId: Int) {
        ratio = args.compressRatios[layerId]
        isCandidateSource = layerId == args.candidateSourceLayerId
        candidateTopkBlocks = args.candidateTopkBlocks
        candidateBlockSize = args.candidateBlockSize
        numHeads = args.indexNHeads
        headDim = args.indexHeadDim
        indexTopk = args.indexTopk
        ropeDim = args.qkRopeHeadDim
        self._qBProj.wrappedValue = Linear(
            args.qLoraRank, args.indexNHeads * args.indexHeadDim, bias: false)
        self._weightsProj.wrappedValue = Linear(args.hiddenSize, args.indexNHeads, bias: false)
        // Only a layer that compresses its own KV makes index keys.
        let makesKeys = args.kvSourceLayerIds.contains(layerId)
        self._kProj.wrappedValue =
            makesKeys ? Linear(args.headDim, args.indexHeadDim, bias: false) : nil
        self._kNorm.wrappedValue =
            makesKeys ? RMSNorm(dimensions: args.indexHeadDim, eps: args.rmsNormEps) : nil
        super.init()
    }

    /// Returns the top-k compressed entries of each query (-1 for none) and the candidate mask:
    /// the one this layer selects, or the one it was given.
    func callAsFunction(
        _ x: MLXArray, qResidual: MLXArray, indexK: MLXArray?, candidates: MLXArray?,
        freqs: [Float], startPos: Int
    ) -> (MLXArray, MLXArray?) {
        let batch = x.dim(0)
        let length = x.dim(1)
        guard let indexK, indexK.dim(1) > 0 else {
            return (MLXArray.zeros([batch, length, 0], type: Int32.self), candidates)
        }
        let keysCount = indexK.dim(1)
        var q = qBProj(qResidual).reshaped(batch, length, numHeads, headDim)
        q = deepseekV41ApplyRope(q, freqs: freqs, ropeDim: ropeDim, offset: startPos)
        let keys = indexK.asType(.float32)
        let weights =
            weightsProj(x).asType(.float32) * Float(pow(Double(numHeads), -0.5))
        var scores = matmul(q.asType(.float32), keys.expandedDimensions(axis: 1).transposed(0, 1, 3, 2))
        scores = maximum(scores, Float(0)) * Float(pow(Double(headDim), -0.5))
        scores = sum(scores * weights.expandedDimensions(axis: -1), axis: 2)  // [B, S, K]

        // A query sees the groups that end at or before it.
        let visibleData: [Bool] = (0 ..< length).flatMap { t -> [Bool] in
            let lens = (startPos + 1 + t) / ratio
            return (0 ..< keysCount).map { $0 < lens }
        }
        let visible = MLXArray(visibleData, [1, length, keysCount])
        scores = which(visible, scores, -Float.infinity)

        var candidatesOut = candidates
        if isCandidateSource && candidateTopkBlocks > 0 {
            let block = candidateBlockSize
            let pad = ((-keysCount) % block + block) % block
            var padded = scores
            if pad > 0 {
                padded = concatenated(
                    [scores, MLXArray.full([batch, length, pad], values: MLXArray(-Float.infinity))],
                    axis: -1)
            }
            let blocks = padded.dim(-1) / block
            var blockScores = padded.reshaped(batch, length, blocks, block).max(axis: -1)
            // Always keep the newest block: it is only partly full, so an older full block can
            // have a higher score.
            let newestData: [Bool] = (0 ..< length).flatMap { t -> [Bool] in
                let lens = (startPos + 1 + t) / ratio
                let last = lens > 0 ? (lens - 1) / block : -1
                return (0 ..< blocks).map { $0 == last }
            }
            blockScores = which(
                MLXArray(newestData, [1, length, blocks]), MLXArray(Float.infinity), blockScores)
            let kBlocks = min(candidateTopkBlocks, blocks)
            let chosen = argPartition(-blockScores, kth: kBlocks - 1, axis: -1)[
                .ellipsis, 0 ..< kBlocks]
            // With fewer reachable blocks than kBlocks, the extra picks are -inf.
            let reachable: MLXArray =
                takeAlong(blockScores, chosen, axis: -1) .> MLXArray(-Float.infinity)
            var blockMask = MLXArray.zeros(blockScores.shape, type: Bool.self)
            blockMask = putAlong(blockMask, chosen, values: reachable, axis: -1)
            candidatesOut = repeated(blockMask, count: block, axis: -1)[
                .ellipsis, 0 ..< scores.dim(-1)]
        } else if let candidates {
            scores = which(candidates, scores, -Float.infinity)
        }

        let k = min(indexTopk, keysCount)
        let lensArray = MLXArray(
            (0 ..< length).map { Int32((startPos + 1 + $0) / ratio) }, [1, length, 1])
        let top = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, 0 ..< k].asType(.int32)
        let valid = which(top .< lensArray, top, MLXArray(Int32(-1)))
        return (valid, candidatesOut)
    }
}

public final class DeepseekV41Attention: Module {
    let numHeads: Int
    let headDim: Int
    let ropeDim: Int
    let oGroups: Int
    let ratio: Int
    let isKVSource: Bool
    let isIndexSource: Bool
    let ropeFreqs: [Float]
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_norm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_proj") var kvProj: Linear
    @ModuleInfo(key: "kv_norm") var kvNorm: RMSNorm
    @ModuleInfo(key: "o_a_proj") var oAProj: DeepseekV41BlockLinear
    @ModuleInfo(key: "o_b_proj") var oBProj: Linear
    @ModuleInfo(key: "compressor") var compressor: DeepseekV41Compressor?
    @ModuleInfo(key: "indexer") var indexer: DeepseekV41Indexer?
    var sinks: MLXArray

    init(_ args: SharpAIDeepseekV41TextConfiguration, layerId: Int) {
        numHeads = args.numAttentionHeads
        headDim = args.headDim
        ropeDim = args.qkRopeHeadDim
        oGroups = args.oGroups
        let ratios = args.compressRatios
        ratio = layerId < ratios.count ? ratios[layerId] : 0
        isKVSource = args.kvSourceLayerIds.contains(layerId)
        isIndexSource = args.indexSourceLayerIds.contains(layerId)
        if ratio > 0 {
            let scaling = args.ropeScaling
            ropeFreqs = deepseekV41RopeFrequencies(
                dims: args.qkRopeHeadDim, base: args.compressRopeTheta,
                factor: scaling?.factor ?? 1.0,
                originalMaxPositionEmbeddings:
                    scaling?.originalMaxPositionEmbeddings ?? args.maxPositionEmbeddings,
                betaFast: scaling?.betaFast ?? 32, betaSlow: scaling?.betaSlow ?? 1)
        } else {
            // A layer with only the window uses the base theta and no YaRN.
            ropeFreqs = deepseekV41RopeFrequencies(
                dims: args.qkRopeHeadDim, base: args.ropeTheta, factor: 1.0,
                originalMaxPositionEmbeddings: 4096, betaFast: 32, betaSlow: 1)
        }
        self._qAProj.wrappedValue = Linear(args.hiddenSize, args.qLoraRank, bias: false)
        self._qANorm.wrappedValue = RMSNorm(dimensions: args.qLoraRank, eps: args.rmsNormEps)
        self._qBProj.wrappedValue = Linear(
            args.qLoraRank, args.numAttentionHeads * args.headDim, bias: false)
        self._kvProj.wrappedValue = Linear(args.hiddenSize, args.headDim, bias: false)
        self._kvNorm.wrappedValue = RMSNorm(dimensions: args.headDim, eps: args.rmsNormEps)
        self._oAProj.wrappedValue = DeepseekV41BlockLinear(
            inputDims: args.numAttentionHeads * args.headDim / args.oGroups,
            outputDims: args.oLoraRank, groups: args.oGroups)
        self._oBProj.wrappedValue = Linear(
            args.oGroups * args.oLoraRank, args.hiddenSize, bias: false)
        self._compressor.wrappedValue =
            isKVSource ? DeepseekV41Compressor(args, layerId: layerId) : nil
        self._indexer.wrappedValue =
            isIndexSource ? DeepseekV41Indexer(args, layerId: layerId) : nil
        self.sinks = MLXArray.zeros([args.numAttentionHeads], type: Float.self)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, startPos: Int, windowIdx: MLXArray, cache: DeepseekV41LayerCache,
        shared: inout [String: MLXArray]
    ) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        let qResidual = qANorm(qAProj(x))
        var q = qBProj(qResidual).reshaped(batch, length, numHeads, headDim)
        q = deepseekV41ApplyRope(q, freqs: ropeFreqs, ropeDim: ropeDim, offset: startPos)

        var kv = kvNorm(kvProj(x))
        kv = deepseekV41ApplyRope(kv, freqs: ropeFreqs, ropeDim: ropeDim, offset: startPos)
        var kvs: [MLXArray] = [cache.updateWindow(kv)]
        var indices: [MLXArray] = [windowIdx]
        if ratio > 0 {
            if isKVSource {
                compress(x, startPos: startPos, cache: cache, shared: &shared)
            }
            if isIndexSource, let indexer {
                let (topk, candidates) = indexer(
                    x, qResidual: qResidual, indexK: shared["index_k"],
                    candidates: shared["candidates"], freqs: ropeFreqs, startPos: startPos)
                shared["topk_idx"] = topk
                if indexer.isCandidateSource, let candidates {
                    shared["candidates"] = candidates
                }
            }
            if let compressed = shared["compress_kv"], let topk = shared["topk_idx"],
                topk.dim(-1) > 0
            {
                kvs.append(compressed)
                indices.append(topk)
            }
        }

        var output = deepseekV41SparseAttention(
            q, kvs: kvs, indices: indices, sinks: sinks,
            scale: Float(pow(Double(headDim), -0.5)))
        output = deepseekV41ApplyRope(
            output, freqs: ropeFreqs, ropeDim: ropeDim, offset: startPos, inverse: true)
        // Group the flattened heads: each output group reads its own contiguous slice.
        let perGroup = numHeads * headDim / oGroups
        let grouped = output.reshaped(batch * length, oGroups, perGroup)
        let projected = oAProj(grouped)
        return oBProj(projected.reshaped(batch, length, -1))
    }

    /// Adds the new compressed KV and index keys to the cache, and shares them with later layers.
    private func compress(
        _ x: MLXArray, startPos: Int, cache: DeepseekV41LayerCache,
        shared: inout [String: MLXArray]
    ) {
        guard let compressor else {
            return
        }
        let (latent, first) = compressor(x, startPos: startPos, cache: cache)
        if let latent {
            let kv = deepseekV41ApplyRope(
                latent, freqs: ropeFreqs, ropeDim: ropeDim, offset: first, scale: Float(ratio))
            cache.compressKV = deepseekV41Append(cache.compressKV, kv)
            if isIndexSource, let indexer, let kProj = indexer.kProj, let kNorm = indexer.kNorm {
                let k = deepseekV41ApplyRope(
                    kNorm(kProj(latent)), freqs: ropeFreqs, ropeDim: ropeDim, offset: first,
                    scale: Float(ratio))
                cache.indexK = deepseekV41Append(cache.indexK, k)
            }
        }
        if let compressKV = cache.compressKV {
            shared["compress_kv"] = compressKV
        }
        if isIndexSource, let indexK = cache.indexK {
            shared["index_k"] = indexK
        }
    }
}

// MARK: - MoE

/// The routed experts (`switch_mlp`): gate, up and down projections over the selected experts.
public final class DeepseekV41SwitchMLP: Module {
    let limit: Float
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        limit = args.swigluLimit
        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: args.hiddenSize, outputDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts, bias: false)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: args.hiddenSize, outputDims: args.moeIntermediateSize,
            numExperts: args.nRoutedExperts, bias: false)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: args.moeIntermediateSize, outputDims: args.hiddenSize,
            numExperts: args.nRoutedExperts, bias: false)
        super.init()
    }

    /// x [N, H], indices [N, k], weights [N, k] -> [N, H] (float32).
    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axes: [-2, -3])  // [N, 1, 1, H]
        let gate = gateProj(expanded, indices)  // [N, k, 1, I]
        let up = upProj(expanded, indices)
        let activated = deepseekV41ClampedSwiGLU(up: up, gate: gate, limit: limit)
        let routed = downProj(activated, indices).squeezed(axis: -2)  // [N, k, H]
        return sum(
            routed.asType(.float32) * weights.expandedDimensions(axis: -1).asType(.float32),
            axis: 1)
    }
}

public final class DeepseekV41Experts: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: DeepseekV41SwitchMLP

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        self._switchMLP.wrappedValue = DeepseekV41SwitchMLP(args)
        super.init()
    }
}

public final class DeepseekV41SharedMLP: Module {
    let limit: Float
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        limit = args.swigluLimit
        self._gateProj.wrappedValue = Linear(
            args.hiddenSize, args.moeIntermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(args.hiddenSize, args.moeIntermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            args.moeIntermediateSize, args.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        return downProj(deepseekV41ClampedSwiGLU(up: up, gate: gate, limit: limit).asType(x.dtype))
    }
}

/// The router: sqrtsoftplus scores, a correction bias for selection only, top-k, and the
/// normalized weights times the routed scaling factor.
public final class DeepseekV41Router: Module {
    let topK: Int
    let scoreFunc: String
    let gateTemp: Float
    let normTopkProb: Bool
    let routeScale: Float
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        topK = args.numExpertsPerTok
        scoreFunc = args.scoringFunc
        gateTemp = args.gateTemp
        normTopkProb = args.normTopkProb
        routeScale = Float(args.routedScalingFactor)
        self._weight.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts, args.hiddenSize], type: Float.self)
        self._eScoreCorrectionBias.wrappedValue = MLXArray.zeros(
            [args.nRoutedExperts], type: Float.self)
        super.init()
    }

    /// x [N, H] -> (indices [N, k], weights [N, k])
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var logits = matmul(x.asType(.float32), weight.asType(.float32).T)
        logits = logits / gateTemp
        let scores: MLXArray
        switch scoreFunc {
        case "sigmoid":
            scores = sigmoid(logits)
        case "softmax":
            scores = softmax(logits, axis: -1)
        default:
            scores = sqrt(softplus(logits))
        }
        let biased = scores + eScoreCorrectionBias
        let chosen = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, 0 ..< topK]
        var weights = takeAlong(scores, chosen, axis: -1)
        if normTopkProb && topK > 1 {
            weights = weights / (sum(weights, axis: -1, keepDims: true) + Float(1e-20))
        }
        return (chosen, weights * routeScale)
    }
}

public final class DeepseekV41MoE: Module {
    let dim: Int
    @ModuleInfo(key: "gate") var gate: DeepseekV41Router
    @ModuleInfo(key: "experts") var experts: DeepseekV41Experts
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV41SharedMLP

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        dim = args.hiddenSize
        self._gate.wrappedValue = DeepseekV41Router(args)
        self._experts.wrappedValue = DeepseekV41Experts(args)
        self._sharedExperts.wrappedValue = DeepseekV41SharedMLP(args)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, dim)
        let (indices, weights) = gate(flat)
        let routed = experts.switchMLP(flat, indices: indices, weights: weights)
        let out = routed + sharedExperts(flat).asType(.float32)
        return out.reshaped(shape).asType(x.dtype)
    }
}

// MARK: - Hyper-connections

/// The hyper-connection (mHC) mix of one sublayer: pre (collapse weights), post (expand gains)
/// and comb (the doubly-stochastic stream mixing).
public final class DeepseekV41HyperConnection: Module {
    let hc: Int
    let iters: Int
    let hcEps: Float
    let normEps: Float
    var fn: MLXArray
    var base: MLXArray
    var scale: MLXArray

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        hc = args.hcMult
        iters = args.hcSinkhornIters
        hcEps = args.hcEps
        normEps = args.rmsNormEps
        let mix = (2 + args.hcMult) * args.hcMult
        fn = MLXArray.zeros([mix, args.hcMult * args.hiddenSize], type: Float.self)
        base = MLXArray.zeros([mix], type: Float.self)
        scale = MLXArray.ones([3], type: Float.self)
        super.init()
    }

    /// streams [B, S, hc, D] -> (pre [B,S,hc], post [B,S,hc], comb [B,S,hc,hc])
    func callAsFunction(_ streams: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let batch = streams.dim(0)
        let length = streams.dim(1)
        var flat = streams.reshaped(batch, length, -1).asType(.float32)
        flat = deepseekV41RMSNormNoWeight(flat, eps: normEps)
        // The scale of each mix column: pre, then post, then comb.
        let scaleIds =
            [Int32](repeating: 0, count: hc) + [Int32](repeating: 1, count: hc)
            + [Int32](repeating: 2, count: hc * hc)
        let mix = matmul(flat, fn.T) * scale[MLXArray(scaleIds)] + base
        let pre = sigmoid(mix[.ellipsis, 0 ..< hc]) + hcEps
        let post = sigmoid(mix[.ellipsis, hc ..< 2 * hc]) * 2
        var comb = mix[.ellipsis, (2 * hc)...].reshaped(batch, length, hc, hc)
        comb = softmax(comb, axis: -1) + hcEps
        return (pre, post, deepseekV41Sinkhorn(comb, iters: iters, eps: hcEps))
    }
}

// MARK: - Decoder

public final class DeepseekV41DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV41Attention
    @ModuleInfo(key: "mlp") var mlp: DeepseekV41MoE
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "attn_hc") var attnHC: DeepseekV41HyperConnection
    @ModuleInfo(key: "ffn_hc") var ffnHC: DeepseekV41HyperConnection

    init(_ args: SharpAIDeepseekV41TextConfiguration, layerId: Int) {
        self._selfAttn.wrappedValue = DeepseekV41Attention(args, layerId: layerId)
        self._mlp.wrappedValue = DeepseekV41MoE(args)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._attnHC.wrappedValue = DeepseekV41HyperConnection(args)
        self._ffnHC.wrappedValue = DeepseekV41HyperConnection(args)
        super.init()
    }

    func callAsFunction(
        _ streams: MLXArray, preMix: MLXArray, startPos: Int, windowIdx: MLXArray,
        cache: DeepseekV41LayerCache, shared: inout [String: MLXArray]
    ) -> (MLXArray, MLXArray) {
        let (attnPre, attnPost, attnComb) = attnHC(streams)
        var h = inputLayernorm(deepseekV41Collapse(streams, preMix))
        h = selfAttn(
            h, startPos: startPos, windowIdx: windowIdx, cache: cache, shared: &shared)
        let afterAttention = deepseekV41Expand(
            h, residual: streams, post: attnPost, comb: attnComb)

        let (ffnPre, ffnPost, ffnComb) = ffnHC(afterAttention)
        let m = mlp(postAttentionLayernorm(deepseekV41Collapse(afterAttention, attnPre)))
        let out = deepseekV41Expand(m, residual: afterAttention, post: ffnPost, comb: ffnComb)
        return (out, ffnPre)
    }
}

public final class DeepseekV41TextModel: Module {
    let args: SharpAIDeepseekV41TextConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV41DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ args: SharpAIDeepseekV41TextConfiguration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map {
            DeepseekV41DecoderLayer(args, layerId: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [DeepseekV41LayerCache]) -> MLXArray {
        let h = embedTokens(inputs)
        let batch = h.dim(0)
        let length = h.dim(1)
        let startPos = cache[0].offset
        // Every layer's window holds the same rows, so all use the same indices.
        let previous = cache[0].window?.dim(1) ?? 0
        let windowIdx = deepseekV41WindowIndices(
            batch: batch, previous: previous, length: length, window: args.slidingWindow)

        var streams = broadcast(
            h.expandedDimensions(axis: 2), to: [batch, length, args.hcMult, args.hiddenSize])
        var preMix = concatenated(
            [
                MLXArray.ones([batch, length, 1], type: Float.self),
                MLXArray.zeros([batch, length, args.hcMult - 1], type: Float.self),
            ], axis: -1)
        var shared: [String: MLXArray] = [:]
        for (layer, layerCache) in zip(layers, cache) {
            (streams, preMix) = layer(
                streams, preMix: preMix, startPos: startPos, windowIdx: windowIdx,
                cache: layerCache, shared: &shared)
        }
        return norm(deepseekV41Collapse(streams, preMix))
    }
}

// MARK: - Model

/// deepseek_v41 text model: the LanguageModel wrapper around `DeepseekV41TextModel`.
public final class SharpAIDeepseekV41Model: Module, LLMModel, LoRAModel {
    public var kvHeads: [Int] {
        Array(repeating: args.numAttentionHeads, count: args.numHiddenLayers)
    }

    let args: SharpAIDeepseekV41TextConfiguration
    public var model: DeepseekV41TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: SharpAIDeepseekV41TextConfiguration) {
        self.args = args
        self.model = DeepseekV41TextModel(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let typed = typedCaches(cache) ?? makeCaches()
        return lmHead(model(inputs, cache: typed))
    }

    /// One `DeepseekV41LayerCache` per layer.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        makeCaches()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIDeepseekV41Sanitizer.toModuleKeys(weights, numExperts: args.nRoutedExperts)
    }

    public var loraLayers: [Module] {
        model.layers
    }

    private func makeCaches() -> [DeepseekV41LayerCache] {
        (0 ..< args.numHiddenLayers).map { _ in
            DeepseekV41LayerCache(windowSize: args.slidingWindow)
        }
    }

    private func typedCaches(_ cache: [KVCache]?) -> [DeepseekV41LayerCache]? {
        guard let cache else {
            return nil
        }
        guard let typed = cache as? [DeepseekV41LayerCache] else {
            fatalError(
                "SharpAIDeepseekV41Model requires DeepseekV41LayerCache caches (from its own "
                    + "newCache(parameters:)); got \(type(of: cache))")
        }
        return typed
    }
}
