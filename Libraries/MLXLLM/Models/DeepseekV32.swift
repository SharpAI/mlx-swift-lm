// Copyright © 2025 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py
// and https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm_moe_dsa.py
//
// DeepSeek V3.2 is V3 plus DeepSeek Sparse Attention (DSA): a "lightning indexer"
// scores every cached key against the query and attention is restricted to the
// top `index_topk` of them. GLM-5.2 (`glm_moe_dsa`) is the same architecture — in
// mlx-lm its model class is a bare `class Model(DSV32Model)` over a differently
// shaped config — so both land here.
//
// STAGE 2 (this file): the indexer is implemented. Below `index_topk` cached
// positions, the reference indexer itself returns no selection —
//
//     if k.shape[2] <= self.index_topk:
//         return None
//
// — so attention stays dense and byte-identical to stage 1 in that regime; this
// is the "inertness" property the stage-2 tests check without needing real
// weights. Past `index_topk` (2048 for GLM-5.2), attention is restricted to the
// indexer's top-k selection.
//
// Absorbed-MLA note: the Python reference's `DeepseekV32Attention` decode path
// gathers only the selected latent/rope entries (`take_along_axis`) because it
// keeps K/V in absorbed per-token latent form until attention time. This port's
// `DeepseekV3Attention`-derived structure materialises full per-head K/V before
// attention (same as stage 1, same as plain V3), so instead of gathering, the
// top-k selection is applied as a boolean mask ANDed with the causal mask —
// masking out non-selected positions in a full attend-over-everything SDPA call
// is numerically identical to attending only over the gathered subset, just
// without the compute/memory saving of not materialising the excluded K/V. This
// also means one code path (not separate prefill/decode paths) is correct for
// both, which is what the tests below check.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV32Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var normTopkProb: Bool
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    /// Kept as a stored property so the synthesised `Encodable` conformance has
    /// somewhere to put the key; `ropeTheta`/`ropeScaling` above are what the model
    /// actually reads.
    var ropeParameters: [String: StringOrNumber]?
    var attentionBias: Bool

    // DSA-specific.
    var indexTopk: Int
    var indexNHeads: Int
    var indexHeadDim: Int
    /// Whether the indexer's own RoPE is "traditional" (rotates adjacent pairs) vs
    /// the default "half-split" scheme. Not necessarily the same as the main
    /// attention's rope, which this port (matching the mlx-lm reference) always
    /// builds with `traditional: true` regardless of this field.
    var indexerRopeInterleave: Bool
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case indexTopk = "index_topk"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 2048
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 0
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 0
        // The reference Indexer hardcodes traditional=True; treat absence the same way.
        indexerRopeInterleave =
            try c.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave) ?? true
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0

        // GLM-5.2 carries no top-level `rope_theta`/`rope_scaling`; both live inside a
        // `rope_parameters` dict, which mlx-lm unpacks in ModelArgs.__post_init__.
        // DeepSeek's own V3.2 configs keep the flat layout, so accept either.
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        self.ropeParameters = ropeParameters

        if let flat = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = flat
        } else if case .float(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = theta
        } else if case .int(let theta) = ropeParameters?["rope_theta"] {
            ropeTheta = Float(theta)
        } else {
            throw DecodingError.keyNotFound(
                CodingKeys.ropeTheta,
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription:
                        "no rope_theta at the top level or inside rope_parameters"))
        }

        if let flat = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) {
            ropeScaling = flat
        } else if let ropeParameters,
            case .string(let type)? = ropeParameters["rope_type"], type != "default"
        {
            // Only forward rope_parameters as scaling when it actually describes a
            // scheme; "default" means plain RoPE and a non-nil dict here would make
            // the V3 stack look for yarn fields that are not there.
            ropeScaling = ropeParameters
        } else {
            ropeScaling = nil
        }
    }

    /// The V3.2 base stack is V3: same MLA projections, same grouped `noaux_tc` gate,
    /// same MoE block. Only the indexer is new, and stage 1 does not build it — so the
    /// model is constructed straight from the equivalent V3 configuration.
    var v3Configuration: DeepseekV3Configuration {
        DeepseekV3Configuration(
            vocabSize: vocabSize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            moeIntermediateSize: moeIntermediateSize,
            numHiddenLayers: numHiddenLayers,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            nSharedExperts: nSharedExperts,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: routedScalingFactor,
            kvLoraRank: kvLoraRank,
            qLoraRank: qLoraRank,
            qkRopeHeadDim: qkRopeHeadDim,
            vHeadDim: vHeadDim,
            qkNopeHeadDim: qkNopeHeadDim,
            normTopkProb: normTopkProb,
            nGroup: nGroup,
            topkGroup: topkGroup,
            numExpertsPerTok: numExpertsPerTok,
            moeLayerFreq: moeLayerFreq,
            firstKDenseReplace: firstKDenseReplace,
            maxPositionEmbeddings: maxPositionEmbeddings,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            attentionBias: attentionBias
        )
    }
}

/// The DSA "lightning indexer". Scores every cached key against the query with a
/// cheap dedicated head set (`index_n_heads` heads of `index_head_dim`, typically
/// much smaller than the main attention heads) and returns the `index_topk`
/// highest-scoring cached positions — or `nil` when the cache isn't yet longer
/// than `index_topk`, matching the reference's "nothing to select yet" regime.
///
/// RoPE here rotates only the first `qk_rope_head_dim` of each `index_head_dim`
/// vector — a partial rotary embedding, same trick `DeepseekV3Attention` doesn't
/// need (its rope-carrying half is a separate tensor) but the indexer's combined
/// per-head vector does. `RoPE(dimensions:)` in mlx-swift already leaves any
/// dimensions past `dimensions` untouched, so this is a plain `initializeRope`
/// call with `dims: qkRopeHeadDim` against a `headDim`-wide tensor — `traditional`
/// comes from `indexer_rope_interleave` (GLM-5.2 ships `true`), independently of
/// the main attention's rope, which this port always builds `traditional: true`
/// regardless (matching the mlx-lm reference's hardcoded value there).
class DeepseekV32Indexer: Module {
    let nHeads: Int
    let headDim: Int
    let indexTopk: Int
    let softmaxScale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    init(_ args: DeepseekV32Configuration) {
        self.nHeads = args.indexNHeads
        self.headDim = args.indexHeadDim
        self.indexTopk = args.indexTopk
        self.softmaxScale = pow(Float(headDim), -0.5)

        self._wqB.wrappedValue = Linear(args.qLoraRank, nHeads * headDim, bias: false)
        self._wk.wrappedValue = Linear(args.hiddenSize, headDim, bias: false)
        self._kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        self._weightsProj.wrappedValue = Linear(args.hiddenSize, nHeads, bias: false)

        self.rope = initializeRope(
            dims: args.qkRopeHeadDim, base: args.ropeTheta,
            traditional: args.indexerRopeInterleave,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    /// - Parameters:
    ///   - x: hidden states, `[B, L, hiddenSize]`.
    ///   - qr: the low-rank query after `q_a_layernorm(q_a_proj(x))` — the same
    ///     intermediate the main attention's `q_b_proj` reads, reused here rather
    ///     than recomputed.
    ///   - causalMask: a materialised boolean causal mask sized to the indexer's
    ///     own cache (`[B, 1, L, cachedLength]`), or `nil` when masking is a no-op
    ///     (decode: a single new query has no future cached position to exclude).
    ///   - cache: the indexer's own key cache (`CacheList`'s second slot) — holds
    ///     only `k`, never queried for `v`.
    /// - Returns: `[B, 1, L, indexTopk]` top-k indices into the cached axis, or
    ///   `nil` if the cache is not yet longer than `indexTopk`.
    func callAsFunction(
        _ x: MLXArray, qr: MLXArray, causalMask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let (b, s, _) = (x.dim(0), x.dim(1), x.dim(2))

        var q = wqB(qr).reshaped(b, s, nHeads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x)
        k = kNorm(k)
        k = k.reshaped(b, 1, s, headDim)

        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)

        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([b, 1, s, 0]))
        }

        guard k.dim(2) > indexTopk else { return nil }

        var scores = matmul(q, k.transposed(0, 1, 3, 2))
        scores = maximum(scores, 0)
        var weights = weightsProj(x) * (pow(Float(nHeads), -0.5) * softmaxScale)
        weights = weights.transposed(0, 2, 1)[.ellipsis, .newAxis]
        scores = scores * weights
        scores = scores.sum(axis: 1, keepDims: true)

        if let causalMask {
            scores = which(causalMask, scores, MLXArray(-Float.infinity))
        }

        return argPartition(-scores, kth: indexTopk - 1, axis: -1)[.ellipsis, ..<indexTopk]
    }
}

/// `DeepseekV3Attention` plus the DSA indexer. The Q/KV projections are an exact
/// duplicate of `DeepseekV3Attention` (same weight keys, same shapes — the
/// checkpoint's main-attention weights are unchanged by DSA); the only new
/// sub-module is `indexer`, and the only behavioral change is building a
/// top-k-restricted mask before the existing attention call. See the file header
/// for why this is a boolean mask rather than the reference's absorbed-MLA gather.
class DeepseekV32Attention: Module {
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    let numHeads: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "indexer") var indexer: DeepseekV32Indexer

    init(_ args: DeepseekV32Configuration) {
        let hiddenSize = args.hiddenSize
        self.numHeads = args.numAttentionHeads
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.vHeadDim = args.vHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.qHeadDim = args.qkNopeHeadDim + args.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        let qLoraRank = args.qLoraRank
        self._qAProj.wrappedValue = Linear(hiddenSize, qLoraRank, bias: args.attentionBias)
        self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
        self._qBProj.wrappedValue = Linear(qLoraRank, numHeads * qHeadDim, bias: false)

        self._kvAProjWithMqa.wrappedValue = Linear(
            hiddenSize, kvLoraRank + qkRopeHeadDim, bias: args.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank, numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim), bias: false)
        self._oProj.wrappedValue = Linear(numHeads * vHeadDim, hiddenSize, bias: args.attentionBias)

        if let ropeScaling = args.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0.0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1.0
                if scalingFactor > 1 {
                    let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1.0
                    self.scale = self.scale * s * s
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim, base: args.ropeTheta, traditional: true,
            scalingConfig: args.ropeScaling, maxPositionEmbeddings: args.maxPositionEmbeddings)
        self._indexer.wrappedValue = DeepseekV32Indexer(args)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let mainCache = cache?[0]
        let indexerCache = cache?[1]

        let qr = qALayerNorm!(qAProj!(x))
        var q = qBProj!(qr)
        q = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var (qNope, qPe) = (splitQ[0], splitQ[1])
        var compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        compressedKv = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kv = kvBProj(kvALayerNorm(compressedKv))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        var (kNope, values) = (splitKv[0], splitKv[1])

        qPe = applyRotaryPosition(rope, to: qPe, cache: mainCache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: mainCache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        var keys: MLXArray
        if let mainCache {
            (keys, values) = mainCache.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }
        let queries = concatenated([qNope, qPe], axis: -1)

        // The indexer needs a materialised boolean causal mask to score against
        // (decode's single new query has nothing causal to exclude, so `nil` is
        // correct and cheap there); the final attention mask stays whatever
        // `mask` already is — `.causal`/`.none` — unless the indexer actually
        // returns a selection, in which case it's replaced with the AND of the
        // two, forcing materialisation only when sparsity is actually active.
        var indexerCausalMask: MLXArray? = nil
        if L > 1 {
            if case .array(let arr) = mask {
                indexerCausalMask = arr
            } else {
                indexerCausalMask = createCausalMask(n: L, offset: mainCache?.offset ?? 0)
            }
        }

        let topkIndices = indexer(x, qr: qr, causalMask: indexerCausalMask, cache: indexerCache)

        var finalMask = mask
        if let topkIndices {
            let sTotal = keys.dim(2)
            var sparseMask = MLXArray.zeros([B, 1, L, sTotal], dtype: .bool)
            sparseMask = putAlong(sparseMask, topkIndices, values: MLXArray(true), axis: -1)
            let causal = indexerCausalMask ?? createCausalMask(n: L, offset: mainCache?.offset ?? 0)
            finalMask = .array(sparseMask & causal)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,  // cache already updated above; a second update would double-append
            scale: scale,
            mask: finalMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class DeepseekV32DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV32Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: DeepseekV32Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV32Attention(args)

        let v3Config = args.v3Configuration
        if v3Config.nRoutedExperts != nil,
            layerIdx >= v3Config.firstKDenseReplace,
            layerIdx % v3Config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV3MoE(config: v3Config)
        } else {
            self.mlp = DeepseekV3MLP(config: v3Config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: CacheList?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class DeepseekV32ModelInner: Module, LayerPartitionable, StreamableMoE {
    var args: DeepseekV32Configuration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV32DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public var gpuLayerCount: Int? = nil
    public var streamExperts: Bool = false
    public var totalLayerCount: Int { layers.count }

    init(_ args: DeepseekV32Configuration) {
        self.args = args
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabSize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.numHiddenLayers).map {
            DeepseekV32DecoderLayer(args, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [CacheList]?) -> MLXArray {
        var h = embedTokens(x)

        // Unlike DeepseekV3ModelInner, the causal mask is not built once and
        // shared across layers: each layer's indexer produces its own top-k
        // selection from its own cache pair, so each layer needs its own
        // (possibly sparsified) mask. `createAttentionMask` here just gives every
        // layer the same cheap symbolic `.causal`/`.none` starting point — the
        // per-layer materialisation only happens inside `DeepseekV32Attention`
        // when that layer's indexer actually returns a selection.
        let attentionMask = createAttentionMask(h: h, cache: cache?.first?[0])

        for (i, layer) in layers.enumerated() {
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts) {
                layer(h, mask: attentionMask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

public class DeepseekV32Model: Module, LLMModel, LoRAModel {
    public var kvHeads: [Int] { Array(repeating: args.numAttentionHeads, count: args.numHiddenLayers) }

    var args: DeepseekV32Configuration
    public var model: DeepseekV32ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV32Configuration) {
        self.args = args
        self.model = DeepseekV32ModelInner(args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache as? [CacheList])
        return lmHead(out)
    }

    /// One `CacheList(main, indexer)` pair per layer — mirrors the reference's
    /// `[CacheList(KVCache(), KVCache()) for _ in self.layers]` and the same
    /// paired-cache shape `FalconH1Model.newCache` uses for its own two-cache
    /// hybrid layers. Both children are `KVCacheSimple`, so `canTrimPromptCache`/
    /// `trimPromptCache` (which require every cache to report `isTrimmable`) keep
    /// working transparently through `CacheList.isTrimmable`/`.trim(_:)`.
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< args.numHiddenLayers).map { _ in CacheList(KVCacheSimple(), KVCacheSimple()) }
    }

    /// Multi-token-prediction layers sit past the main stack and have no module to
    /// load into — same reasoning as `DeepseekV3Model.isMultiTokenPredictionLayer`.
    func isMultiTokenPredictionLayer(_ key: String) -> Bool {
        guard key.starts(with: "model.layers.") else { return false }
        let parts = key.split(separator: ".")
        guard parts.count >= 3, let layerIdx = Int(parts[2]) else { return false }
        return layerIdx >= args.numHiddenLayers
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var newWeights = weights

        func dequant(weight: MLXArray, scaleInv: MLXArray) -> MLXArray {
            let bs = 128
            let (m, n) = (weight.dim(0), weight.dim(1))
            let padBottom = (bs - m % bs) % bs
            let padSide = (bs - n % bs) % bs

            var padded = padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
            padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
            let scaled = padded * scaleInv[0..., .newAxis, 0..., .newAxis]
            return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
        }

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                if let weight = weights[weightKey] {
                    let dequantized = dequant(weight: weight, scaleInv: value)
                    newWeights[weightKey] = dequantized
                }
            } else if newWeights[key] == nil {
                newWeights[key] = value
            }
        }

        for l in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(l)"
            for (_, projName) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    if weights[firstKey] != nil {
                        let joined = (0 ..< (args.nRoutedExperts ?? 1)).map {
                            weights["\(prefix).mlp.experts.\($0).\(projName).\(key)"]!
                        }
                        newWeights["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(joined)
                    }
                }
            }
        }

        // Stage 1 dropped `.indexer.` weights entirely (no module to load them
        // into). Stage 2 has `DeepseekV32Indexer`, so they're kept — only
        // `.attn.compressor.` (an unrelated, still-unimplemented long-range
        // compression feature some checkpoints in this family also carry) is
        // still dropped.
        return newWeights.filter { key, _ in
            !isMultiTokenPredictionLayer(key)
                && !key.contains("rotary_emb.inv_freq")
                && !key.contains(".attn.compressor.")
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
