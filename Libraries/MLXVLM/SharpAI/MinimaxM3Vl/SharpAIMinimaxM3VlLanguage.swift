// SharpAI fork overlay: minimax_m3_vl text backbone. Not part of upstream mlx-swift-lm.
//
// Port of MiniMaxM3VLTextModel and MiniMaxM3VLForCausalLM from
// huggingface/transformers src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py
// (main branch, 2026). Reference classes: MiniMaxM3VLAttention, MiniMaxM3VLIndexer,
// MiniMaxM3VLSparseMoeBlock, MiniMaxM3VLTopKRouter, MiniMaxM3VLExperts, MiniMaxM3VLDenseMLP,
// MiniMaxM3VLRMSNorm (Gemma-style, weight + 1), MiniMaxM3VLRotaryEmbedding (partial RoPE).
//
// Scope: the text decoder only, one full-sequence forward with no KV cache.
// Not implemented here: the vision tower, the patch merger, the multimodal projector,
// MTP layers (the reference skips them on load too), and the block-sparse kernel. Sparse
// layers use the reference eager path: the indexer picks key blocks and they become a dense
// additive mask over the causal mask.
// UNVERIFIED: not compiled or run yet. The forward test in the registration tests is the first check.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Text-backbone hyperparameters. Field meanings follow `MiniMaxM3VLTextConfig` in transformers.
public struct SharpAIMinimaxM3VlTextConfiguration: Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    /// Fraction of `headDim` that receives RoPE (`partial_rotary_factor`, 0.5 in M3).
    public var partialRotaryFactor: Float
    public var vocabularySize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var numLocalExperts: Int
    public var numExpertsPerTok: Int
    /// Routed expert intermediate size (`intermediate_size`).
    public var moeIntermediateSize: Int
    public var denseIntermediateSize: Int
    public var sharedIntermediateSize: Int
    public var swigluAlpha: Float
    public var swigluLimit: Float
    public var routedScalingFactor: Float
    public var indexNumHeads: Int
    public var indexHeadDim: Int
    public var indexBlockSize: Int
    public var indexTopkBlocks: Int
    public var indexLocalBlocks: Int
    /// Per layer: `"full_attention"` or `"minimax_m3_sparse"`.
    public var layerTypes: [String]
    /// Per layer: `"dense"` or `"sparse"` (MoE).
    public var mlpLayerTypes: [String]

    public init(
        hiddenSize: Int,
        numHiddenLayers: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        partialRotaryFactor: Float,
        vocabularySize: Int,
        rmsNormEps: Float,
        ropeTheta: Float,
        numLocalExperts: Int,
        numExpertsPerTok: Int,
        moeIntermediateSize: Int,
        denseIntermediateSize: Int,
        sharedIntermediateSize: Int,
        swigluAlpha: Float,
        swigluLimit: Float,
        routedScalingFactor: Float,
        indexNumHeads: Int,
        indexHeadDim: Int,
        indexBlockSize: Int,
        indexTopkBlocks: Int,
        indexLocalBlocks: Int,
        layerTypes: [String],
        mlpLayerTypes: [String]
    ) {
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.partialRotaryFactor = partialRotaryFactor
        self.vocabularySize = vocabularySize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.moeIntermediateSize = moeIntermediateSize
        self.denseIntermediateSize = denseIntermediateSize
        self.sharedIntermediateSize = sharedIntermediateSize
        self.swigluAlpha = swigluAlpha
        self.swigluLimit = swigluLimit
        self.routedScalingFactor = routedScalingFactor
        self.indexNumHeads = indexNumHeads
        self.indexHeadDim = indexHeadDim
        self.indexBlockSize = indexBlockSize
        self.indexTopkBlocks = indexTopkBlocks
        self.indexLocalBlocks = indexLocalBlocks
        self.layerTypes = layerTypes
        self.mlpLayerTypes = mlpLayerTypes
    }

    var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }
}

// MARK: - Shared helpers

/// Gemma-style RMSNorm: normalizes in float32 and scales by `1 + weight` (weight is zero-initialized).
class SharpAIMinimaxM3VlRMSNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let normed = xf * rsqrt((xf * xf).mean(axis: -1, keepDims: true) + eps)
        return (normed * (1 + weight.asType(.float32))).asType(dtype)
    }
}

/// `rotate_half` over the last axis.
func sharpAIMinimaxM3VlRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// Partial RoPE: rotates the first `cos.dim(-1)` channels of the last axis, passes the rest through.
/// `x` is `[B, H, S, D]`; `cos` and `sin` are `[S, R]`.
func sharpAIMinimaxM3VlApplyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let r = cos.dim(-1)
    let c = cos.reshaped([1, 1, cos.dim(0), r])
    let s = sin.reshaped([1, 1, sin.dim(0), r])
    let xRot = x[.ellipsis, ..<r]
    let xPass = x[.ellipsis, r...]
    let rotated = xRot * c + sharpAIMinimaxM3VlRotateHalf(xRot) * s
    return concatenated([rotated, xPass], axis: -1)
}

/// Default RoPE tables for positions `0 ..< seqLen`, rotary width `rotaryDim` (`cat(freqs, freqs)`).
func sharpAIMinimaxM3VlRotaryTables(seqLen: Int, rotaryDim: Int, theta: Float) -> (MLXArray, MLXArray) {
    let half = rotaryDim / 2
    let invFreq: [Float] = (0 ..< half).map { i in 1.0 / pow(theta, Float(2 * i) / Float(rotaryDim)) }
    var emb = [Float]()
    emb.reserveCapacity(seqLen * rotaryDim)
    for pos in 0 ..< seqLen {
        for j in 0 ..< rotaryDim {
            emb.append(Float(pos) * invFreq[j % half])
        }
    }
    let table = MLXArray(emb, [seqLen, rotaryDim])
    return (MLX.cos(table), MLX.sin(table))
}

/// SwiGLU-OAI: the gate is clamped above, the up branch is clamped to ±limit, and the output is `(up + 1) * gate * sigmoid(alpha * gate)`.
/// The gate and up halves are the first and second halves of the last axis.
func sharpAIMinimaxM3VlSwigluOAI(_ gateUp: MLXArray, limit: Float, alpha: Float) -> MLXArray {
    let half = gateUp.dim(-1) / 2
    let gate = clip(gateUp[.ellipsis, ..<half], max: limit)
    let up = clip(gateUp[.ellipsis, half...], min: -limit, max: limit)
    let glu = gate * sigmoid(gate * alpha)
    return (up + 1) * glu
}

// MARK: - Attention and lightning indexer

/// Lightning indexer for sparse layers. Scores each query against every key with a small
/// multi-head branch, max-pools the scores into key blocks of `blockSize`, keeps the top
/// `topkBlocks` blocks per query plus the `localBlocks` blocks at and before the query's block,
/// and returns the resulting key-level keep mask.
class SharpAIMinimaxM3VlIndexer: Module {
    let numHeads: Int
    let headDim: Int
    let blockSize: Int
    let topkBlocks: Int
    let localBlocks: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3VlRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3VlRMSNorm

    init(_ config: SharpAIMinimaxM3VlTextConfiguration) {
        self.numHeads = config.indexNumHeads
        self.headDim = config.indexHeadDim
        self.blockSize = config.indexBlockSize
        self.topkBlocks = config.indexTopkBlocks
        self.localBlocks = config.indexLocalBlocks

        _qProj.wrappedValue = Linear(
            config.hiddenSize, config.indexNumHeads * config.indexHeadDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, config.indexHeadDim, bias: false)
        _qNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: config.indexHeadDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: config.indexHeadDim, eps: config.rmsNormEps)
    }

    /// Returns a boolean key-level mask `[B, numAttentionHeads, S, S]`: true where the query may
    /// read the key through the block selection. Causality is not applied here; the caller ANDs it in.
    /// Each indexer head selects for its own group of `numAttentionHeads / numHeads` query heads.
    func keepMask(_ hidden: MLXArray, cos: MLXArray, sin: MLXArray, numAttentionHeads: Int) -> MLXArray {
        let batch = hidden.dim(0)
        let seq = hidden.dim(1)

        var idxQ = qNorm(qProj(hidden).reshaped([batch, seq, numHeads, headDim])).transposed(0, 2, 1, 3)
        var idxK = kNorm(kProj(hidden).reshaped([batch, seq, 1, headDim])).transposed(0, 2, 1, 3)
        idxQ = sharpAIMinimaxM3VlApplyRotary(idxQ, cos: cos, sin: sin)
        idxK = sharpAIMinimaxM3VlApplyRotary(idxK, cos: cos, sin: sin)

        let numBlocks = (seq + blockSize - 1) / blockSize
        let pad = numBlocks * blockSize - seq

        // Position-only masks are built on the host: they depend only on `seq`.
        var futureFlags = [Bool](repeating: false, count: seq * seq)
        var localFlags = [Bool](repeating: false, count: seq * numBlocks)
        for q in 0 ..< seq {
            let queryBlock = q / blockSize
            for k in 0 ..< seq where k > q {
                futureFlags[q * seq + k] = true
            }
            for l in 0 ..< localBlocks {
                let block = max(queryBlock - l, 0)
                localFlags[q * numBlocks + block] = true
            }
        }
        let futureMask = MLXArray(futureFlags, [1, 1, seq, seq])
        let localMask = MLXArray(localFlags, [1, 1, seq, numBlocks])

        // Token-level scores, with future keys removed.
        var scores = matmul(idxQ.asType(.float32), idxK.asType(.float32).transposed(0, 1, 3, 2))
        let negInf = MLXArray(-Float.infinity)
        scores = MLX.where(futureMask, negInf, scores)
        if pad > 0 {
            let padding = MLXArray.zeros([batch, numHeads, seq, pad]) + negInf
            scores = concatenated([scores, padding], axis: -1)
        }

        // Max-pool each block of keys to one score per (query, block).
        let blockScores = scores
            .reshaped([batch, numHeads, seq, numBlocks, blockSize])
            .max(axis: -1)

        // The local blocks always win their slots.
        let boosted = MLX.where(localMask, MLXArray(Float.infinity), blockScores)

        let topk = min(topkBlocks, numBlocks)
        let picked = argPartition(-boosted, kth: topk - 1, axis: -1)[.ellipsis, ..<topk]

        // Per (query, indexer head, block): selected if any of the top-k slots names it.
        let blockIds = MLXArray((0 ..< numBlocks).map { UInt32($0) })
        let hits = MLX.equal(picked.expandedDimensions(axis: -1), blockIds)  // [B, Hi, S, topk, NB]
        let blockKeep = hits.any(axis: -2)  // [B, Hi, S, NB]

        // Expand each block verdict onto its keys, then onto the query heads of its group.
        let keyKeep = repeated(blockKeep, count: blockSize, axis: -1)[.ellipsis, ..<seq]
        return repeated(keyKeep, count: numAttentionHeads / numHeads, axis: 1)
    }
}

class SharpAIMinimaxM3VlAttention: Module {
    let numHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3VlRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3VlRMSNorm
    @ModuleInfo(key: "indexer") var indexer: SharpAIMinimaxM3VlIndexer?

    init(_ config: SharpAIMinimaxM3VlTextConfiguration, layerIndex: Int) {
        self.numHeads = config.numAttentionHeads
        self.numKeyValueHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, numKeyValueHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(
            numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        let isSparse = config.layerTypes[layerIndex] == "minimax_m3_sparse"
        _indexer.wrappedValue = isSparse ? SharpAIMinimaxM3VlIndexer(config) : nil
    }

    /// `causal` is a boolean `[1, 1, S, S]` mask.
    func callAsFunction(
        _ hidden: MLXArray, cos: MLXArray, sin: MLXArray, causal: MLXArray
    ) -> MLXArray {
        let batch = hidden.dim(0)
        let seq = hidden.dim(1)

        var q = qNorm(qProj(hidden).reshaped([batch, seq, numHeads, headDim])).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(hidden).reshaped([batch, seq, numKeyValueHeads, headDim]))
            .transposed(0, 2, 1, 3)
        let v = vProj(hidden).reshaped([batch, seq, numKeyValueHeads, headDim]).transposed(0, 2, 1, 3)

        q = sharpAIMinimaxM3VlApplyRotary(q, cos: cos, sin: sin)
        k = sharpAIMinimaxM3VlApplyRotary(k, cos: cos, sin: sin)

        let groups = numHeads / numKeyValueHeads
        let kFull = repeated(k, count: groups, axis: 1)
        let vFull = repeated(v, count: groups, axis: 1)

        var scores = matmul(q, kFull.transposed(0, 1, 3, 2)) * scale
        let negInf = MLXArray(-Float.infinity)
        scores = MLX.where(causal, scores, negInf)
        if let indexer {
            let blockKeep = indexer.keepMask(hidden, cos: cos, sin: sin, numAttentionHeads: numHeads)
            scores = MLX.where(blockKeep, scores, negInf)
        }

        let weights = softmax(scores.asType(.float32), axis: -1).asType(q.dtype)
        let out = matmul(weights, vFull)
            .transposed(0, 2, 1, 3)
            .reshaped([batch, seq, numHeads * headDim])
        return oProj(out)
    }
}

// MARK: - Feed-forward (dense or MoE)

/// Router with sigmoid scores. The correction bias only chooses experts; the mixing weights use the raw sigmoid scores, renormalized over the chosen experts.
class SharpAIMinimaxM3VlRouter: Module {
    let topK: Int
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var scoreCorrectionBias: MLXArray

    init(_ config: SharpAIMinimaxM3VlTextConfiguration) {
        self.topK = config.numExpertsPerTok
        _weight.wrappedValue = MLXArray.zeros([config.numLocalExperts, config.hiddenSize])
        _scoreCorrectionBias.wrappedValue = MLXArray.zeros([config.numLocalExperts])
    }

    /// `x` is `[N, H]`. Returns the mixing weights and expert indices, both `[N, topK]`.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(weight.dtype), weight.transposed())
        let routing = sigmoid(logits.asType(.float32))
        let choice = routing + scoreCorrectionBias.asType(.float32)
        let indices = argPartition(-choice, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var mixing = takeAlong(routing, indices, axis: -1)
        mixing = mixing / mixing.sum(axis: -1, keepDims: true)
        return (mixing, indices)
    }
}

/// Routed experts stored as stacked 3-D tensors: `gate_up_proj [E, 2I, H]`, `down_proj [E, H, I]`.
class SharpAIMinimaxM3VlExperts: Module {
    let numExperts: Int
    let limit: Float
    let alpha: Float

    @ParameterInfo(key: "gate_up_proj") var gateUpProj: MLXArray
    @ParameterInfo(key: "down_proj") var downProj: MLXArray

    init(_ config: SharpAIMinimaxM3VlTextConfiguration) {
        self.numExperts = config.numLocalExperts
        self.limit = config.swigluLimit
        self.alpha = config.swigluAlpha
        _gateUpProj.wrappedValue = MLXArray.zeros([
            config.numLocalExperts, 2 * config.moeIntermediateSize, config.hiddenSize,
        ])
        _downProj.wrappedValue = MLXArray.zeros([
            config.numLocalExperts, config.hiddenSize, config.moeIntermediateSize,
        ])
    }

    /// `x` is `[N, H]`; `indices` and `weights` are `[N, topK]`. Runs each expert on all tokens and
    /// zeroes the tokens that did not route to it, which matches the reference result.
    func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
        var out = x * Float(0)
        for e in 0 ..< numExperts {
            let hit = MLX.equal(indices, MLXArray(UInt32(e)))
            let w = MLX.where(hit, weights, MLXArray(Float(0))).sum(axis: -1, keepDims: true)
            let gateUp = matmul(x, gateUpProj[e].transposed())
            let act = sharpAIMinimaxM3VlSwigluOAI(gateUp, limit: limit, alpha: alpha)
            let y = matmul(act, downProj[e].transposed())
            out = out + y * w
        }
        return out
    }
}

/// Feed-forward of one decoder layer. Dense layers use `gate_up_proj` / `down_proj` directly.
/// Sparse layers use `gate`, `experts`, and `shared_experts`, whose weights are at the checkpoint keys.
class SharpAIMinimaxM3VlFeedForward: Module {
    let isSparse: Bool
    let limit: Float
    let alpha: Float
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear?
    @ModuleInfo(key: "down_proj") var downProj: Linear?
    @ModuleInfo(key: "gate") var router: SharpAIMinimaxM3VlRouter?
    @ModuleInfo(key: "experts") var experts: SharpAIMinimaxM3VlExperts?
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIMinimaxM3VlFeedForward?

    /// Dense MLP (`denseIntermediate`) or a sparse MoE block with one shared expert.
    init(_ config: SharpAIMinimaxM3VlTextConfiguration, isSparse: Bool, denseIntermediate: Int? = nil) {
        self.isSparse = isSparse
        self.limit = config.swigluLimit
        self.alpha = config.swigluAlpha
        self.routedScalingFactor = config.routedScalingFactor

        if isSparse {
            _gateUpProj.wrappedValue = nil
            _downProj.wrappedValue = nil
            _router.wrappedValue = SharpAIMinimaxM3VlRouter(config)
            _experts.wrappedValue = SharpAIMinimaxM3VlExperts(config)
            _sharedExperts.wrappedValue = SharpAIMinimaxM3VlFeedForward(
                config, sharedIntermediate: config.sharedIntermediateSize)
        } else {
            let inter = denseIntermediate ?? config.denseIntermediateSize
            _gateUpProj.wrappedValue = Linear(config.hiddenSize, 2 * inter, bias: false)
            _downProj.wrappedValue = Linear(inter, config.hiddenSize, bias: false)
            _router.wrappedValue = nil
            _experts.wrappedValue = nil
            _sharedExperts.wrappedValue = nil
        }
    }

    /// Init for the shared expert, a dense SwiGLU-OAI MLP with `sharedIntermediate` width.
    private init(_ config: SharpAIMinimaxM3VlTextConfiguration, sharedIntermediate: Int) {
        self.isSparse = false
        self.limit = config.swigluLimit
        self.alpha = config.swigluAlpha
        self.routedScalingFactor = config.routedScalingFactor
        _gateUpProj.wrappedValue = Linear(config.hiddenSize, 2 * sharedIntermediate, bias: false)
        _downProj.wrappedValue = Linear(sharedIntermediate, config.hiddenSize, bias: false)
        _router.wrappedValue = nil
        _experts.wrappedValue = nil
        _sharedExperts.wrappedValue = nil
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        if !isSparse {
            let act = sharpAIMinimaxM3VlSwigluOAI(gateUpProj!(hidden), limit: limit, alpha: alpha)
            return downProj!(act)
        }

        let batch = hidden.dim(0)
        let seq = hidden.dim(1)
        let dim = hidden.dim(2)
        let flat = hidden.reshaped([batch * seq, dim])

        let shared = sharedExperts!(flat)
        let (mixing, indices) = router!(flat)
        let routed = experts!(flat, indices: indices, weights: mixing) * routedScalingFactor
        return (routed + shared).reshaped([batch, seq, dim])
    }
}

// MARK: - Decoder

class SharpAIMinimaxM3VlDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinimaxM3VlAttention
    @ModuleInfo(key: "mlp") var mlp: SharpAIMinimaxM3VlFeedForward
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIMinimaxM3VlRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: SharpAIMinimaxM3VlRMSNorm

    init(_ config: SharpAIMinimaxM3VlTextConfiguration, layerIndex: Int) {
        _selfAttn.wrappedValue = SharpAIMinimaxM3VlAttention(config, layerIndex: layerIndex)
        _mlp.wrappedValue = SharpAIMinimaxM3VlFeedForward(
            config, isSparse: config.mlpLayerTypes[layerIndex] == "sparse")
        _inputLayerNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, causal: MLXArray
    ) -> MLXArray {
        var h = x + selfAttn(inputLayerNorm(x), cos: cos, sin: sin, causal: causal)
        h = h + mlp(postAttentionLayerNorm(h))
        return h
    }
}

/// `MiniMaxM3VLTextModel`: embeddings, decoder stack, and the final norm.
class SharpAIMinimaxM3VlTextModel: Module {
    let configuration: SharpAIMinimaxM3VlTextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIMinimaxM3VlDecoderLayer]
    @ModuleInfo(key: "norm") var norm: SharpAIMinimaxM3VlRMSNorm

    init(_ config: SharpAIMinimaxM3VlTextConfiguration) {
        self.configuration = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            SharpAIMinimaxM3VlDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = SharpAIMinimaxM3VlRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seq = inputIds.dim(1)
        var h = embedTokens(inputIds)

        let (cos, sin) = sharpAIMinimaxM3VlRotaryTables(
            seqLen: seq, rotaryDim: configuration.rotaryDim, theta: configuration.ropeTheta)

        var causalFlags = [Bool](repeating: false, count: seq * seq)
        for q in 0 ..< seq {
            for k in 0 ... q {
                causalFlags[q * seq + k] = true
            }
        }
        let causal = MLXArray(causalFlags, [1, 1, seq, seq])

        for layer in layers {
            h = layer(h, cos: cos, sin: sin, causal: causal)
        }
        return norm(h)
    }
}

/// `MiniMaxM3VLForCausalLM` for the text backbone. Positions are `0 ..< S`, and there is no KV cache.
public class SharpAIMinimaxM3VlForCausalLM: Module {
    @ModuleInfo(key: "model") var model: SharpAIMinimaxM3VlTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: SharpAIMinimaxM3VlTextConfiguration) {
        _model.wrappedValue = SharpAIMinimaxM3VlTextModel(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    /// `inputIds` is `[B, S]` of token ids. Returns logits `[B, S, vocabularySize]`.
    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        lmHead(model(inputIds))
    }
}
