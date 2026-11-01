//
//  SharpAIHyV4Attention.swift
//  MLXLLM
//
//  Multi-head latent attention with a per-head sigmoid output gate, learnable
//  attention sinks, and the DSA top-k indexer for hy_v4 (HYV4ForCausalLM).
//  Port of HYV4Attention and HYV4Indexer in transformers modular_hy_v4.py.
//
//  A "full" layer runs the indexer and selects `index_topk` keys per query. A
//  "shared" layer reuses the selection of the most recent full layer in the same
//  forward pass. The reference passes top-k indices and scatters them into the
//  mask. This port passes the boolean selection mask instead, which gives the
//  same attention.
//
//  Attention is computed explicitly (not with MLXFast) because the sink logits
//  are appended as an extra softmax column, matching eager_attention_forward.
//
//  UNVERIFIED: checkpoint key names (`self_attn.indexer.*`, `self_attn.gate_proj`,
//  `self_attn.sinks`) follow the transformers module names. No hy_v4 safetensors
//  index is on disk, so they are not checked against a real checkpoint.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Top-k key selection (HYV4Indexer). Keys for previously cached positions are kept in an
/// indexer-only `KVCacheSimple` that is tied to the owning attention cache.
class SharpAIHyV4Indexer: Module {
    let headCount: Int
    let headDim: Int
    let ropeDim: Int
    let topK: Int
    let scoreScale: Float
    let headScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    let rope: RoPELayer

    /// Indexer keys for the owning cache, [B, 1, T, headDim]. Not a module parameter.
    private var keyCache: KVCacheSimple?
    private weak var keyCacheOwner: AnyObject?

    init(_ config: SharpAIHyV4Configuration) {
        self.headCount = config.resolvedIndexHeads
        self.headDim = config.resolvedIndexHeadDim
        self.ropeDim = config.resolvedRopeHeadDim
        self.topK = config.resolvedIndexTopK
        self.scoreScale = pow(Float(config.resolvedIndexHeadDim), -0.5)
        self.headScale = pow(Float(config.resolvedIndexHeads), -0.5)

        _wqB.wrappedValue = Linear(
            config.resolvedQLoraRank, headCount * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: config.rmsNormEps)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, headCount, bias: false)

        self.rope = initializeRope(
            dims: ropeDim,
            base: config.resolvedRopeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Boolean selection mask [B, L, T]. True marks the keys each query selects.
    /// The caller still applies the causal mask: masked future keys can tie at -inf.
    func selection(
        _ x: MLXArray, qResid: MLXArray, offset: RoPEOffset?, owner: KVCache?, causal: MLXArray
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let nopeDim = headDim - ropeDim

        // Rope applies to the trailing `ropeDim` channels, as in the reference.
        let qParts = split(
            wqB(qResid).reshaped(B, L, headCount, headDim).transposed(0, 2, 1, 3),
            indices: [nopeDim], axis: -1)
        let kParts = split(
            kNorm(wk(x)).reshaped(B, L, 1, headDim).transposed(0, 2, 1, 3),
            indices: [nopeDim], axis: -1)
        let q = concatenated(
            [qParts[0], applyRotaryPosition(rope, to: qParts[1], offset: offset)], axis: -1)
        let k = concatenated(
            [kParts[0], applyRotaryPosition(rope, to: kParts[1], offset: offset)], axis: -1)
        let keys = cachedKeys(k, owner: owner)  // [B, 1, T, headDim]

        // index_score[b, s, t] = sum_h w[b, s, h] * relu(q[b, h, s] . k[b, t])
        let logits = relu(
            matmul(q.asType(.float32), keys.asType(.float32).transposed(0, 1, 3, 2)))
        let headWeights = (weightsProj(x).asType(.float32) * (headScale * scoreScale))
            .transposed(0, 2, 1)
            .expandedDimensions(axis: -1)  // [B, H, L, 1]
        let scores = (logits * headWeights).sum(axis: 1)  // [B, L, T]
        let masked = MLX.where(causal, scores, MLXArray(-Float.infinity))

        let T = keys.dim(2)
        let kept = min(topK, T)
        let top = argPartition(-masked, kth: kept - 1, axis: -1)[.ellipsis, ..<kept]
        let none = MLXArray.zeros(masked.shape, type: Bool.self)
        return putAlong(none, top, values: MLXArray(true), axis: -1)
    }

    /// Indexer keys for every position in `owner`. The cache is rebuilt when the owner
    /// changes or when its offset no longer lines up with the indexer's own length.
    private func cachedKeys(_ k: MLXArray, owner: KVCache?) -> MLXArray {
        guard let owner else { return k }
        let ownerObject = owner as AnyObject
        // The owning cache has already advanced by this call's tokens.
        let priorLength = owner.offset - k.dim(2)
        if keyCache == nil || keyCacheOwner !== ownerObject || keyCache!.offset != priorLength {
            keyCache = KVCacheSimple()
            keyCacheOwner = ownerObject
        }
        return keyCache!.update(keys: k, values: k).0
    }
}

/// Latent attention with sinks, output gate, and optional DSA selection.
class SharpAIHyV4Attention: Module {
    let numHeads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let qkHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray
    @ModuleInfo(key: "indexer") var indexer: SharpAIHyV4Indexer?

    let rope: RoPELayer

    init(_ config: SharpAIHyV4Configuration, hasIndexer: Bool) {
        self.numHeads = config.attentionHeads
        self.qkNopeHeadDim = config.resolvedNopeHeadDim
        self.qkRopeHeadDim = config.resolvedRopeHeadDim
        self.qkHeadDim = config.resolvedNopeHeadDim + config.resolvedRopeHeadDim
        self.kvLoraRank = config.resolvedKvLoraRank
        self.vHeadDim = config.resolvedVHeadDim
        // yarn_apply_mscale is the identity for the default rope type.
        self.scale = pow(Float(qkHeadDim), -0.5)

        let hidden = config.hiddenSize
        _qAProj.wrappedValue = Linear(hidden, config.resolvedQLoraRank, bias: config.attentionBias ?? false)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.resolvedQLoraRank, eps: config.rmsNormEps)
        _qBProj.wrappedValue = Linear(config.resolvedQLoraRank, numHeads * qkHeadDim, bias: false)
        _kvAProjWithMqa.wrappedValue = Linear(
            hidden, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias ?? false)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: config.rmsNormEps)
        _kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false)
        _gateProj.wrappedValue = Linear(hidden, numHeads * vHeadDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * vHeadDim, hidden, bias: config.attentionBias ?? false)
        // learnable_sink_init defaults to 0.0 in HYV4Config.
        _sinks.wrappedValue = zeros([numHeads])
        _indexer.wrappedValue = hasIndexer ? SharpAIHyV4Indexer(config) : nil

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: config.resolvedRopeTheta,
            traditional: false,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    /// Returns the attention output and the selection mask to pass to the next layer.
    /// `causal` is [L, T] boolean. `sparse` is the selection from the last full layer
    /// and is used only when this layer has no indexer.
    func callAsFunction(
        _ x: MLXArray, causal: MLXArray, sparse: MLXArray?, cache: KVCache?
    ) -> (output: MLXArray, selection: MLXArray?) {
        let (B, L) = (x.dim(0), x.dim(1))
        let offset = cache?.ropeOffset

        let qResid = qALayerNorm(qAProj(x))
        let qStates = qBProj(qResid).reshaped(B, L, numHeads, qkHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(qStates, indices: [qkNopeHeadDim], axis: -1)
        let qPass = qParts[0]
        let qRot = applyRotaryPosition(rope, to: qParts[1], offset: offset)

        let compressed = kvAProjWithMqa(x)
        let kvParts = split(compressed, indices: [kvLoraRank], axis: -1)
        let latent = kvALayerNorm(kvParts[0]).expandedDimensions(axis: 1)  // [B, 1, L, R]
        let kRot = applyRotaryPosition(
            rope,
            to: kvParts[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3),
            offset: offset)  // [B, 1, L, rope]

        // The cache holds the compressed latent and the shared rope key, one head each.
        var allLatent = latent
        var allRot = kRot
        if let cache {
            let stored = cache.update(
                keys: concatenated([latent, kRot], axis: -1), values: latent)
            let storedKeys = split(stored.0, indices: [kvLoraRank], axis: -1)
            allRot = storedKeys[1]
            allLatent = stored.1
        }
        let T = allLatent.dim(2)

        // Expand the latent into per-head keys and values (expand_kv).
        let expanded = kvBProj(MLX.squeezed(allLatent, axis: 1))
            .reshaped(B, T, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let expandedParts = split(expanded, indices: [qkNopeHeadDim], axis: -1)
        let values = expandedParts[1]  // [B, H, T, V]
        let keys = concatenated(
            [expandedParts[0], repeated(allRot, count: numHeads, axis: 1)], axis: -1)  // [B, H, T, qk]
        let queries = concatenated([qPass, qRot], axis: -1)  // [B, H, L, qk]

        let selection: MLXArray
        if let indexer {
            selection = indexer.selection(
                x, qResid: qResid, offset: offset, owner: cache, causal: causal)
        } else {
            guard let sparse else {
                preconditionFailure(
                    "hy_v4: a shared DSA layer needs the selection of an earlier full layer")
            }
            selection = sparse
        }

        let allowed = logicalAnd(selection.expandedDimensions(axis: 1), causal)  // [B, 1, L, T]
        var scores = matmul(
            queries.asType(.float32), keys.asType(.float32).transposed(0, 1, 3, 2)) * scale
        scores = MLX.where(allowed, scores, MLXArray(-Float.infinity))

        // Sink logits join the softmax as one extra column, which is dropped afterwards.
        let sinkLogits = broadcast(
            sinks.asType(.float32).reshaped(1, numHeads, 1, 1), to: [B, numHeads, L, 1])
        let probs = softmax(concatenated([scores, sinkLogits], axis: -1), axis: -1)
        let weights = split(probs, indices: [T], axis: -1)[0]
        let attended = matmul(weights.asType(values.dtype), values)  // [B, H, L, V]

        let gate = gateProj(x).reshaped(B, L, numHeads, vHeadDim).transposed(0, 2, 1, 3)
        let gated = attended * sigmoid(gate).asType(attended.dtype)
        let output = gated.transposed(0, 2, 1, 3).reshaped(B, L, -1).asType(x.dtype)
        return (oProj(output), selection)
    }
}
