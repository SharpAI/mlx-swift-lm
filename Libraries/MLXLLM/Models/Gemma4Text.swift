//
//  Gemma4Text.swift
//  mlx-swift-lm
//
// Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gemma4_text.py
//
// Gemma 4 text backbone.
// Architecture highlights:
//   - 30 layers with alternating sliding-window / full attention (pattern: 5 sliding + 1 full)
//   - Sliding attention : head_dim=256, 8 KV heads, local RoPE (theta=10 000), window=1024
//   - Full attention    : head_dim=512, 2 KV heads, global RoPE (theta=1 000 000, 25% partial),
//                         attention_k_eq_v (values = raw k_proj output, v_norm_no_scale applied)
//   - Every layer has BOTH a shared dense MLP and a sparse MoE expert block (outputs summed)
//   - MoE: 128 experts, top-8, moe_intermediate_size=704, GeGLU activation
//   - Per-layer learnable scalar (layer_scalar) applied to residual output
//   - Embedding scale: sqrt(hidden_size)
//   - Final logit soft-capping: tanh(x / 30) * 30

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int       // shared dense MLP hidden dim
    var moeIntermediateSize: Int    // sparse expert hidden dim
    var attentionHeads: Int         // Q heads (same for all layers)
    var headDim: Int                // sliding-attention head dim
    var globalHeadDim: Int          // full-attention head dim
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int                // KV heads for sliding layers
    var globalKvHeads: Int          // KV heads for full-attention layers
    var slidingWindow: Int
    var maxPositionEmbeddings: Int
    var attentionKEqV: Bool         // full-attn: values = raw k_proj output
    var finalLogitSoftcapping: Float
    var enableMoeBlock: Bool
    var numExperts: Int
    var topKExperts: Int
    var tieWordEmbeddings: Bool
    var layerTypes: [String]        // "sliding_attention" | "full_attention"
    var ropeParameters: [String: [String: StringOrNumber]]?

    // Derived rope values
    var slidingRopeTheta: Float {
        ropeParameters?["sliding_attention"]?["rope_theta"]?.asFloat() ?? 10_000.0
    }
    var fullRopeTheta: Float {
        ropeParameters?["full_attention"]?["rope_theta"]?.asFloat() ?? 1_000_000.0
    }
    var fullPartialRotaryFactor: Float {
        ropeParameters?["full_attention"]?["partial_rotary_factor"]?.asFloat() ?? 0.25
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case globalKvHeads = "num_global_key_value_heads"
        case slidingWindow = "sliding_window"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKEqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    enum OuterKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        // Support both VLM wrapper (outer key "text_config") and flat layout
        let outerContainer = try decoder.container(keyedBy: OuterKeys.self)
        let c: KeyedDecodingContainer<CodingKeys>
        if outerContainer.contains(.textConfig) {
            c = try outerContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
        } else {
            c = try decoder.container(keyedBy: CodingKeys.self)
        }

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        globalHeadDim = try c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? headDim
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        globalKvHeads = try c.decodeIfPresent(Int.self, forKey: .globalKvHeads) ?? kvHeads
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        attentionKEqV = try c.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        finalLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        enableMoeBlock = try c.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        topKExperts = try c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 0
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        ropeParameters = try c.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        if let lt = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = lt
        } else {
            // Generate from default 5+1 pattern when not specified in config
            var types = [String]()
            for i in 0..<hiddenLayers {
                types.append(i % 6 == 5 ? "full_attention" : "sliding_attention")
            }
            layerTypes = types
        }
    }
}

// MARK: - Helpers

/// RMS normalisation without a learnable scale weight.
/// Equivalent to Python's RMSNormNoScale (used for v_norm in full-attention layers).
private func rmsNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let rms = rsqrt(x.square().mean(axis: -1, keepDims: true) + eps)
    return x * rms
}

// MARK: - Attention

/// Attention block supporting both sliding-window and full-attention variants.
class Gemma4Attention: Module {
    let isSliding: Bool
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let useKEqV: Bool
    let slidingWindow: Int
    let eps: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?   // absent for full-attn with k_eq_v
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Gemma.RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Gemma.RMSNorm

    let rope: RoPE

    init(_ config: Gemma4TextConfiguration, isSliding: Bool) {
        self.isSliding = isSliding
        self.nHeads = config.attentionHeads
        self.slidingWindow = config.slidingWindow
        self.eps = config.rmsNormEps

        let dim = config.hiddenSize

        if isSliding {
            self.nKVHeads = config.kvHeads
            self.headDim = config.headDim
            self.useKEqV = false

            _qProj.wrappedValue = Linear(dim, config.attentionHeads * config.headDim, bias: false)
            _kProj.wrappedValue = Linear(dim, config.kvHeads * config.headDim, bias: false)
            _vProj.wrappedValue = Linear(dim, config.kvHeads * config.headDim, bias: false)
            _oProj.wrappedValue = Linear(config.attentionHeads * config.headDim, dim, bias: false)
            _qNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = Gemma.RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            self.rope = RoPE(
                dimensions: config.headDim, traditional: false, base: config.slidingRopeTheta)
        } else {
            self.nKVHeads = config.globalKvHeads
            self.headDim = config.globalHeadDim
            self.useKEqV = config.attentionKEqV

            _qProj.wrappedValue = Linear(
                dim, config.attentionHeads * config.globalHeadDim, bias: false)
            _kProj.wrappedValue = Linear(
                dim, config.globalKvHeads * config.globalHeadDim, bias: false)
            if !config.attentionKEqV {
                _vProj.wrappedValue = Linear(
                    dim, config.globalKvHeads * config.globalHeadDim, bias: false)
            }
            _oProj.wrappedValue = Linear(
                config.attentionHeads * config.globalHeadDim, dim, bias: false)
            _qNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.globalHeadDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = Gemma.RMSNorm(
                dimensions: config.globalHeadDim, eps: config.rmsNormEps)
            let ropeDims = Int(Float(config.globalHeadDim) * config.fullPartialRotaryFactor)
            self.rope = RoPE(
                dimensions: ropeDims, traditional: false, base: config.fullRopeTheta)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let offset = cache?.offset ?? 0

        // Queries: project → q_norm → transpose → RoPE
        var queries = qProj(x).reshaped(B, L, nHeads, headDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)
        queries = rope(queries, offset: offset)

        // Keys: project → (save raw for k_eq_v values) → k_norm → transpose → RoPE
        let rawKeys = kProj(x).reshaped(B, L, nKVHeads, headDim)

        var values: MLXArray
        if useKEqV {
            // Values = v_norm_no_scale applied to raw keys (before k_norm)
            values = rmsNormNoScale(rawKeys, eps: eps).transposed(0, 2, 1, 3)
        } else {
            values = vProj!(x).reshaped(B, L, nKVHeads, headDim).transposed(0, 2, 1, 3)
        }

        var keys = kNorm(rawKeys).transposed(0, 2, 1, 3)
        keys = rope(keys, offset: offset)

        // Scale = 1.0 for all Gemma 4 attention layers
        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: 1.0,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - Shared Dense MLP

/// Shared dense MLP present in every decoder layer (GeGLU activation).
class Gemma4MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Expert Router

/// Routes tokens to top-k experts with renormalisation and per-expert scaling.
class Gemma4Router: Module {
    /// Learnable per-hidden-dim scale (shape: [hidden_size])
    var scale: MLXArray
    /// Learnable per-expert output scale (shape: [num_experts])
    var perExpertScale: MLXArray
    @ModuleInfo(key: "proj") var proj: Linear

    let eps: Float
    let rootSize: Float  // hidden_size^{-0.5}
    let topK: Int

    init(_ config: Gemma4TextConfiguration) {
        self.eps = config.rmsNormEps
        self.rootSize = pow(Float(config.hiddenSize), -0.5)
        self.topK = config.topKExperts
        self.scale = MLXArray.ones([config.hiddenSize])
        self.perExpertScale = MLXArray.ones([config.numExperts])
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // RMSNorm without learnable weight
        var h = rmsNormNoScale(x, eps: eps)
        h = h * rootSize
        h = h * scale

        let expertScores = proj(h)
        let routerProbs = softmax(expertScores, axis: -1, precise: true)

        let inds = argPartition(-expertScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        var weights = takeAlong(routerProbs, inds, axis: -1)
        weights = weights / weights.sum(axis: -1, keepDims: true)

        // Apply per-expert output scale: perExpertScale[inds]
        let flatInds = inds.flattened()
        let expertWeights = perExpertScale[flatInds].reshaped(inds.shape)
        weights = weights * expertWeights

        return (inds, weights)
    }
}

// MARK: - Sparse Expert Block

/// Wraps SwitchGLU with the key path expected by saved weights ("experts.switch_glu.*").
class Gemma4Experts: Module {
    @ModuleInfo(key: "switch_glu") var switchGlu: SwitchGLU

    init(_ config: Gemma4TextConfiguration) {
        _switchGlu.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
            activation: geluApproximate
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, _ topKIndices: MLXArray, _ topKWeights: MLXArray
    ) -> MLXArray {
        // x: [B, S, H]  topKIndices/topKWeights: [B, S, K]
        let expertOut = switchGlu(x, topKIndices)  // → [B, S, K, H]
        return (expertOut * topKWeights.expandedDimensions(axis: -1)).sum(axis: -2)  // → [B, S, H]
    }
}

// MARK: - Decoder Layer

class Gemma4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma.RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma.RMSNorm

    // MoE-only norms (nil when enableMoeBlock = false)
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: Gemma.RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: Gemma.RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: Gemma.RMSNorm?

    // MoE router and experts (nil when enableMoeBlock = false)
    @ModuleInfo(key: "router") var router: Gemma4Router?
    @ModuleInfo(key: "experts") var experts: Gemma4Experts?

    /// Learnable per-layer residual scalar (shape: [1])
    var layerScalar: MLXArray

    let enableMoe: Bool

    init(_ config: Gemma4TextConfiguration, isSliding: Bool) {
        self.enableMoe = config.enableMoeBlock
        self.layerScalar = MLXArray.ones([1])

        _selfAttn.wrappedValue = Gemma4Attention(config, isSliding: isSliding)
        _mlp.wrappedValue = Gemma4MLP(
            dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        _inputLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _preFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postFeedforwardLayerNorm.wrappedValue = Gemma.RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.enableMoeBlock {
            _postFeedforwardLayerNorm1.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _preFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postFeedforwardLayerNorm2.wrappedValue = Gemma.RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _router.wrappedValue = Gemma4Router(config)
            _experts.wrappedValue = Gemma4Experts(config)
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        // Attention sub-layer
        let attnOut = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        var h = x + postAttentionLayerNorm(attnOut)

        // Feed-forward sub-layer
        let residual = h
        let ffOut: MLXArray
        if enableMoe,
            let norm1 = postFeedforwardLayerNorm1,
            let preNorm2 = preFeedforwardLayerNorm2,
            let norm2 = postFeedforwardLayerNorm2,
            let rtr = router,
            let exp = experts
        {
            // Shared dense path
            let h1 = norm1(mlp(preFeedforwardLayerNorm(h)))

            // Sparse MoE path
            let (topKIndices, topKWeights) = rtr(h)
            let h2 = norm2(exp(preNorm2(h), topKIndices, topKWeights))

            ffOut = postFeedforwardLayerNorm(h1 + h2)
        } else {
            ffOut = postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        }
        h = residual + ffOut

        // Per-layer residual scalar
        h = h * layerScalar

        return h
    }
}

// MARK: - Inner Model

public class Gemma4Model: Module, LayerPartitionable, StreamableMoE {
    // LayerPartitionable
    public var gpuLayerCount: Int?
    public var totalLayerCount: Int { layers.count }

    // StreamableMoE
    public var streamExperts: Bool = false

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [Gemma4DecoderLayer]
    let norm: Gemma.RMSNorm
    let config: Gemma4TextConfiguration

    init(_ config: Gemma4TextConfiguration) {
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)

        self.layers = (0..<config.hiddenLayers).map { i in
            Gemma4DecoderLayer(config, isSliding: config.layerTypes[i] == "sliding_attention")
        }

        self.norm = Gemma.RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        // Scale embeddings by sqrt(hidden_size)
        let embeddingScale = MLXArray(sqrt(Float(config.hiddenSize))).asType(h.dtype)
        h = h * embeddingScale

        // Build the two attention masks: one for global full-attention, one for sliding window
        let firstFullIdx = config.layerTypes.firstIndex(of: "full_attention")
        let firstSlidingIdx = config.layerTypes.firstIndex(of: "sliding_attention")

        let globalMask = createAttentionMask(
            h: h,
            cache: firstFullIdx.flatMap { cache?[$0] })
        let slidingMask = createAttentionMask(
            h: h,
            cache: firstSlidingIdx.flatMap { cache?[$0] },
            windowSize: config.slidingWindow)

        for (i, layer) in layers.enumerated() {
            let mask = config.layerTypes[i] == "full_attention" ? globalMask : slidingMask
            h = partitionedLayerCall(
                index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts
            ) {
                layer(h, mask: mask, cache: cache?[i])
            }
        }

        return norm(h)
    }
}

// MARK: - Top-level LLM Model

public class Gemma4TextModel: Module, LLMModel {
    public var vocabularySize: Int { config.vocabularySize }

    @ModuleInfo public var model: Gemma4Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4TextConfiguration

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.model = Gemma4Model(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, cache: cache)

        if let lmHead {
            out = lmHead(out)
        } else {
            // Tied embeddings: use the embedding matrix as the projection
            out = model.embedTokens.asLinear(out)
        }

        // Final logit soft-capping: tanh(x / cap) * cap
        let cap = config.finalLogitSoftcapping
        out = tanh(out / cap) * cap

        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var w = weights

        // Strip language_model prefix (VLM-converted weights)
        let unflat = ModuleParameters.unflattened(weights)
        if let lm = unflat["language_model"] {
            w = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        // Remove lm_head when embeddings are tied
        if config.tieWordEmbeddings {
            w["lm_head.weight"] = nil
            w["lm_head.scales"] = nil
            w["lm_head.biases"] = nil
        }

        // Trim over-sized vocab tensors (e.g. padded to a multiple of 256)
        let vocabKeys = [
            "model.embed_tokens.weight",
            "model.embed_tokens.scales",
            "model.embed_tokens.biases",
        ]
        for key in vocabKeys {
            if let t = w[key], t.dim(0) > config.vocabularySize {
                w[key] = t[0..<config.vocabularySize]
            }
        }

        // If HF BF16 weights arrive with per-expert stacked format
        // (experts.gate_up_proj / experts.down_proj), split them into the
        // SwitchGLU layout (experts.switch_glu.{gate,up,down}_proj.weight).
        // HF format: gate_up_proj [num_experts, 2*moe_hidden, hidden_size]
        //            down_proj    [num_experts, hidden_size, moe_hidden]
        for l in 0..<config.hiddenLayers {
            let prefix = "model.layers.\(l)"

            if let gateUp = w.removeValue(forKey: "\(prefix).experts.gate_up_proj") {
                let parts = split(gateUp, indices: [gateUp.dim(1) / 2], axis: 1)
                w["\(prefix).experts.switch_glu.gate_proj.weight"] = parts[0]
                w["\(prefix).experts.switch_glu.up_proj.weight"] = parts[1]
            }

            if let down = w.removeValue(forKey: "\(prefix).experts.down_proj") {
                w["\(prefix).experts.switch_glu.down_proj.weight"] = down
            }
        }

        return w
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        (0..<config.hiddenLayers).map { i in
            if config.layerTypes[i] == "full_attention" {
                let cache = StandardKVCache()
                cache.step = 1024
                return cache as KVCache
            } else {
                return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
            }
        }
    }
}

// MARK: - LoRA

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
