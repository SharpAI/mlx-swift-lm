//
//  SharpAIMinimaxM3TextModel.swift
//  MLXLLM
//
//  SharpAI fork overlay: text decoder for model_type minimax_m3_text. Not part of upstream
//  mlx-swift-lm.
//
//  Pre-norm decoder. Layers before index 3 use a dense SwiGLU-OAI MLP. Later layers use a
//  sigmoid-routed MoE: top-k is chosen on score + bias, weights come from the unbiased scores
//  renormalized over the chosen experts, the routed sum is scaled by routed_scaling_factor, and
//  the shared expert is added after. Attention is GQA with per-head Gemma-style qk RMSNorm and
//  partial RoPE. A sparse layer also runs a block indexer. It max-pools query-key scores per
//  block, always keeps the query's local block, keeps the top-k blocks per indexer head group,
//  and intersects that block set with causal visibility.
//
//  Each layer keeps a CacheList(main KV, indexer key history). The indexer keys carry RoPE, so
//  they cannot be rebuilt from the main cache.
//
//  UNVERIFIED (no M3 checkpoint run here):
//  - Indexer RoPE covers the full index_dim, and its norms are Gemma-style. The reference
//    summary says the cos/sin tables are sliced to 128 channels, which fits full-width RoPE.
//  - Expert and shared-expert key layout (fused gate_up_proj vs per-expert w1/w2/w3 or
//    gate/up/down_proj). SharpAIMinimaxM3TextSanitizer.remapCheckpoint accepts all of them.
//  - The shared expert width is shared_intermediate_size times n_shared_experts.
//  - The unbiased-score renormalization in the router comes from the Laguna sibling
//    (renormalize over the chosen experts, no epsilon). It is not checked against M3.
//  The routing, activation constants and sparse-selection rules follow the transformers
//  minimax_m3_vl modular file and config as summarized, not as byte-level source.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Gemma-style RMSNorm: normalizes in float32 and scales by (1 + weight). The weight starts at 0.
class SharpAIMinimaxM3TextRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight + 1, eps: eps).asType(x.dtype)
    }
}

/// SwiGLU-OAI: g = min(gate, limit), out = (clip(up, -limit, limit) + 1) * (g * sigmoid(alpha * g)).
func sharpAIMinimaxM3SwigluOAI(gate: MLXArray, up: MLXArray, alpha: Float, limit: Float)
    -> MLXArray
{
    let g = clip(gate, max: limit)
    let u = clip(up, min: -limit, max: limit)
    return (u + 1) * (g * sigmoid(g * alpha))
}

/// Dense SwiGLU-OAI MLP. Used by the dense layers and by the shared expert.
class SharpAIMinimaxM3TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let alpha: Float
    let limit: Float

    init(hiddenSize: Int, intermediateSize: Int, alpha: Float, limit: Float) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        self.alpha = alpha
        self.limit = limit
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(
            sharpAIMinimaxM3SwigluOAI(
                gate: gateProj(x), up: upProj(x), alpha: alpha, limit: limit))
    }
}

/// Sigmoid router with a selection-only bias. Checkpoint keys: `mlp.gate.weight` and
/// `mlp.gate.e_score_correction_bias`.
class SharpAIMinimaxM3TextRouter: Module {
    let topK: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(hiddenSize: Int, numExperts: Int, topK: Int) {
        self.topK = topK
        _weight.wrappedValue = MLXArray.zeros([numExperts, hiddenSize])
        _eScoreCorrectionBias.wrappedValue = MLXArray.zeros([numExperts])
        super.init()
    }

    /// Returns expert indices [B, L, topK] and renormalized unbiased sigmoid weights [B, L, topK].
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = x.matmul(weight.T).asType(.float32)
        let scores = sigmoid(logits)
        let selection = scores + eScoreCorrectionBias
        let inds = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, inds, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)
        return (inds, weights.asType(x.dtype))
    }
}

/// Routed experts (SwitchGLU with the SwiGLU-OAI activation) plus an optional shared expert.
class SharpAIMinimaxM3TextSparseMoE: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: SharpAIMinimaxM3TextRouter
    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SharpAIMinimaxM3TextMLP?
    let routedScalingFactor: Float

    init(_ config: SharpAIMinimaxM3TextConfiguration) {
        let alpha = config.swigluAlpha
        let limit = config.swigluLimit
        self.routedScalingFactor = config.routedScalingFactor

        _gate.wrappedValue = SharpAIMinimaxM3TextRouter(
            hiddenSize: config.hiddenSize,
            numExperts: config.numLocalExperts ?? 0,
            topK: config.numExpertsPerTok ?? 1)

        _experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize ?? 0,
            numExperts: config.numLocalExperts ?? 0,
            activation: { g in
                let clamped = clip(g, max: limit)
                return clamped * sigmoid(clamped * alpha)
            },
            activationProduct: { g, u in
                sharpAIMinimaxM3SwigluOAI(gate: g, up: u, alpha: alpha, limit: limit)
            },
            bias: false)

        let sharedWidth =
            config.numSharedExperts * (config.sharedIntermediateSize ?? config.intermediateSize ?? 0)
        if config.numSharedExperts > 0, sharedWidth > 0 {
            _sharedExperts.wrappedValue = SharpAIMinimaxM3TextMLP(
                hiddenSize: config.hiddenSize, intermediateSize: sharedWidth,
                alpha: alpha, limit: limit)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = gate(x)
        let expertOutputs = experts(x, inds)
        var routed = (expertOutputs * weights[.ellipsis, .newAxis]).sum(axis: -2)
        routed = routed * routedScalingFactor
        if let sharedExperts {
            routed = routed + sharedExperts(x)
        }
        return routed
    }
}

/// Block indexer for sparse layers. It returns, for each query, the key blocks it may attend
/// to, one set per indexer head. The cache holds this layer's indexer key history.
class SharpAIMinimaxM3TextIndexer: Module {
    let numHeads: Int
    let headDim: Int
    let blockSize: Int
    let topkBlocks: Int
    let localBlocks: Int
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3TextConfiguration, sparse: SharpAIMinimaxM3TextSparseAttentionConfiguration) {
        self.numHeads = sparse.indexHeads
        self.headDim = sparse.indexDim
        self.blockSize = sparse.blockSize
        self.topkBlocks = sparse.topkBlocks
        self.localBlocks = sparse.localBlocks
        self.rope = RoPE(dimensions: sparse.indexDim, traditional: false, base: config.ropeTheta)

        _qProj.wrappedValue = Linear(config.hiddenSize, sparse.indexHeads * sparse.indexDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, sparse.indexDim, bias: false)
        _qNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: sparse.indexDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: sparse.indexDim, eps: config.rmsNormEps)
        super.init()
    }

    /// - Returns: a boolean block mask `[B, numHeads, L, blockCount]` (true = block kept for that
    ///   query), and the key length after this step's append (`cache offset + L`).
    func callAsFunction(_ x: MLXArray, cache: KVCache?) -> (MLXArray, Int) {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = qNorm(qProj(x).reshaped(B, L, numHeads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, 1, headDim)).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        q = applyRotaryPosition(rope, to: q, offset: offset)
        k = applyRotaryPosition(rope, to: k, offset: offset)
        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([B, 1, L, 0]))
        }

        let keyLength = k.dim(2)
        let blockCount = (keyLength + blockSize - 1) / blockSize
        let paddedLength = blockCount * blockSize
        let negInf = MLXArray(-Float.infinity)

        // Raw dot products, no 1/sqrt(d) scale. Future keys are masked before pooling.
        var scores = matmul(q.asType(.float32), k.asType(.float32).transposed(0, 1, 3, 2))
        scores = which(createCausalMask(n: L, offset: keyLength - L), scores, negInf)
        if paddedLength > keyLength {
            scores = padded(
                scores,
                widths: [
                    .init((0, 0)), .init((0, 0)), .init((0, 0)),
                    .init((0, paddedLength - keyLength)),
                ],
                value: negInf)
        }
        var blockScores = scores.reshaped(B, numHeads, L, blockCount, blockSize).max(axis: -1)

        // Local blocks: the query's own block, then localBlocks - 1 before it (clamped at 0).
        // They always win the selection.
        let queryPositions = MLXArray((keyLength - L ..< keyLength).map { Int32($0) })
        let queryBlocks = floorDivide(queryPositions, MLXArray(Int32(blockSize)))
        let blockIndex = MLXArray((0 ..< blockCount).map { Int32($0) })
        var isLocal = MLXArray.zeros([L, blockCount], dtype: .bool)
        for o in 0 ..< localBlocks {
            let target = maximum(queryBlocks - MLXArray(Int32(o)), MLXArray(Int32(0)))
            isLocal = isLocal | (blockIndex.reshaped(1, blockCount) .== target.reshaped(L, 1))
        }
        blockScores = which(isLocal, MLXArray(Float.infinity), blockScores)

        // Top-k blocks per query and indexer head. Blocks that are still -inf are not selected.
        let keep = min(topkBlocks, blockCount)
        let order = argPartition(-blockScores, kth: keep - 1, axis: -1)[.ellipsis, ..<keep]
        let picked = takeAlong(blockScores, order, axis: -1)
        let valid = picked .> negInf
        // Invalid slots go to a dummy column past the last block, which is dropped below.
        let slots = which(valid, order.asType(.int32), MLXArray(Int32(blockCount)))
        var selected = MLXArray.zeros([B, numHeads, L, blockCount + 1], dtype: .bool)
        selected = putAlong(selected, slots, values: MLXArray(true), axis: -1)
        return (selected[.ellipsis, ..<blockCount], keyLength)
    }
}

/// GQA attention with per-head qk RMSNorm, partial RoPE, and an optional block indexer.
class SharpAIMinimaxM3TextAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: SharpAIMinimaxM3TextRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: SharpAIMinimaxM3TextRMSNorm?
    @ModuleInfo(key: "indexer") var indexer: SharpAIMinimaxM3TextIndexer?

    init(_ config: SharpAIMinimaxM3TextConfiguration, layer: Int) {
        self.numHeads = config.attentionHeads
        self.numKVHeads = config.kvHeads
        self.headDim = config.resolvedHeadDim
        self.scale = pow(Float(config.resolvedHeadDim), -0.5)
        self.rope = RoPE(
            dimensions: config.rotaryDimensions, traditional: false, base: config.ropeTheta)

        _qProj.wrappedValue = Linear(
            config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(
            numHeads * headDim, config.hiddenSize, bias: false)

        if config.useQkNorm {
            _qNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
                dimensions: headDim, eps: config.rmsNormEps)
        }

        if config.isSparseAttentionLayer(layer), let sparse = config.sparseAttention {
            _indexer.wrappedValue = SharpAIMinimaxM3TextIndexer(config, sparse: sparse)
        }
        super.init()
    }

    /// - Parameter cache: `CacheList(main KV, indexer keys)` for this layer, or nil.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let mainCache = cache?[0]

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim)
        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = mainCache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var attentionMask = mask
        if let indexer {
            // The indexer appends to its own cache first. Its key length equals the main
            // cache's offset plus L, so the block mask is sized for the main attention below.
            let (blockMask, keyLength) = indexer(x, cache: cache?[1])
            let blockOfKey = MLXArray((0 ..< keyLength).map { Int32($0 / indexer.blockSize) })
            let keyVisible = take(blockMask, blockOfKey, axis: -1)
            let causal = createCausalMask(n: L, offset: keyLength - L)
                .reshaped(1, 1, L, keyLength)
            let visible = keyVisible & causal
            // Query heads in one KV group share one indexer head.
            attentionMask = .array(
                repeated(visible, count: numHeads / indexer.numHeads, axis: 1))
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: mainCache,
            scale: scale,
            mask: attentionMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class SharpAIMinimaxM3TextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SharpAIMinimaxM3TextAttention
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: SharpAIMinimaxM3TextRMSNorm
    @ModuleInfo(key: "post_attention_layernorm")
    var postAttentionLayerNorm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3TextConfiguration, layer: Int) {
        _selfAttn.wrappedValue = SharpAIMinimaxM3TextAttention(config, layer: layer)

        if config.isMoELayer(layer) {
            _mlp.wrappedValue = SharpAIMinimaxM3TextSparseMoE(config)
        } else {
            _mlp.wrappedValue = SharpAIMinimaxM3TextMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.denseWidth ?? 0,
                alpha: config.swigluAlpha,
                limit: config.swigluLimit)
        }

        _inputLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class SharpAIMinimaxM3TextInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [SharpAIMinimaxM3TextDecoderLayer]
    let norm: SharpAIMinimaxM3TextRMSNorm

    init(_ config: SharpAIMinimaxM3TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            SharpAIMinimaxM3TextDecoderLayer(config, layer: $0)
        }
        self.norm = SharpAIMinimaxM3TextRMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [CacheList]?) -> MLXArray {
        var h = embedTokens(inputs)
        // Dense layers take this mask. Sparse layers build their own from the block selection.
        let mask = createAttentionMask(h: h, cache: cache?.first?[0])
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

class SharpAIMinimaxM3TextModel: Module, LLMModel {
    let vocabularySize: Int
    let kvHeads: [Int]
    let configuration: SharpAIMinimaxM3TextConfiguration
    let model: SharpAIMinimaxM3TextInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ configuration: SharpAIMinimaxM3TextConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
        self.model = SharpAIMinimaxM3TextInner(configuration)

        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize, configuration.vocabularySize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: Self.typedCaches(cache))
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// One `CacheList(main, indexer)` per layer. Each child is a `KVCacheSimple`. Bounded
    /// (rotating) caches are not supported here, so `maxKVSize` is ignored.
    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ -> KVCache in
            CacheList(KVCacheSimple(), KVCacheSimple())
        }
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        SharpAIMinimaxM3TextSanitizer.remapCheckpoint(weights, configuration: configuration)
    }

    private static func typedCaches(_ cache: [KVCache]?) -> [CacheList]? {
        guard let cache else { return nil }
        let typed = cache.compactMap { $0 as? CacheList }
        precondition(
            typed.count == cache.count,
            "minimax_m3_text expects one CacheList per layer, as produced by newCache")
        return typed
    }
}

extension SharpAIMinimaxM3TextModel: LoRAModel {
    var loraLayers: [Module] {
        model.layers
    }
}
