//
//  Gemma4.swift
//  mlx-swift-lm
//
//  Created for SwiftLM Gemma 4 Support
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Specialized norm for Gemma (weights are stored as offsets from 1.0)
public class Gemma4RMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    public init(dimensions: Int, eps: Float = 1e-5) {
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // vLLM: standard RMSNorm — weight * x / sqrt(E[x^2] + eps)
        // Gemma 4 norm weights are trained scale factors (~8-10), NOT zero-init offsets.
        return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

public class Gemma4RMSNormNoScale: Module {
    let eps: Float

    public init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xFloat32 = x.asType(.float32)
        let meanSq = MLX.mean(MLX.square(xFloat32), axes: [-1], keepDims: true)
        let inverseNorm = MLX.rsqrt(meanSq + eps)
        return (xFloat32 * inverseNorm).asType(x.dtype)
    }
}

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Matches the Python `ProportionalRoPE` from `rope_utils.py`:
/// frequencies are computed for the first `rotated_dims` dimensions,
/// padded with `inf` for the remainder so the kernel leaves them unchanged.
public class Gemma4ProportionalRoPE: Module, OffsetLayer {
    let dims: Int
    let rotatedDims: Int
    let traditional: Bool
    let theta: Float

    public init(dims: Int, traditional: Bool = false, theta: Float = 10000.0,
                partialRotaryFactor: Float = 1.0, factor: Float = 1.0) {
        self.dims = dims
        self.rotatedDims = Int(Float(dims) * partialRotaryFactor)
        self.traditional = traditional
        self.theta = theta
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        if rotatedDims == dims {
            // Full rotation — standard RoPE with dims as frequency denominator
            return MLXFast.RoPE(
                x, dimensions: dims, traditional: traditional,
                base: theta, scale: 1.0, offset: offset, freqs: nil)
        }
        
        // Partial rotation (proportional RoPE for Gemma4 global attention)
        // vLLM gemma4_rope.py: freq_exponents use head_size (full dims) as denominator
        // For Neox-style (traditional = false), x is implicitly split into two halves.
        // We must rotate the FIRST rope_angles of the first half, and the FIRST rope_angles of the second half.
        if traditional {
            // Interleaved style
            let x1 = x[0..., 0..., 0..., 0..<rotatedDims]
            let x2 = x[0..., 0..., 0..., rotatedDims...]
            
            let rotatedX1 = MLXFast.RoPE(
                x1, dimensions: dims, traditional: traditional,
                base: theta, scale: 1.0, offset: offset, freqs: nil)
                
            return concatenated([rotatedX1, x2], axis: -1)
        } else {
            // Neox style
            let halfDim = dims / 2
            let ropeAngles = rotatedDims / 2
            
            // Extract the dimensions to be rotated
            let xRope1 = x[0..., 0..., 0..., 0..<ropeAngles]
            let xRope2 = x[0..., 0..., 0..., halfDim..<(halfDim + ropeAngles)]
            let xRope = concatenated([xRope1, xRope2], axis: -1)
            
            // Apply RoPE with Neox style. Pass FULL dims as frequency denominator.
            let rotatedXRope = MLXFast.RoPE(
                xRope, dimensions: dims, traditional: false,
                base: theta, scale: 1.0, offset: offset, freqs: nil)
            
            let rotatedXRope1 = rotatedXRope[0..., 0..., 0..., 0..<ropeAngles]
            let rotatedXRope2 = rotatedXRope[0..., 0..., 0..., ropeAngles...]
            
            // Extract the untouched dimensions
            let xNope1 = x[0..., 0..., 0..., ropeAngles..<halfDim]
            let xNope2 = x[0..., 0..., 0..., (halfDim + ropeAngles)...]
            
            // Concatenate everything back
            let newX1 = concatenated([rotatedXRope1, xNope1], axis: -1)
            let newX2 = concatenated([rotatedXRope2, xNope2], axis: -1)
            
            return concatenated([newX1, newX2], axis: -1)
        }
    }
}

public struct Gemma4VisionConfigurationProxy: Codable, Sendable {
    public let hiddenSize: Int?
    public let hiddenLayers: Int?
    public let intermediateSize: Int?
    public let attentionHeads: Int?
    public let patchSize: Int?
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
    }
}

public struct Gemma4AudioConfigurationProxy: Codable, Sendable {
    public let modelType: String?
    public let hiddenSize: Int?
    public let numHiddenLayers: Int?
    public let numAttentionHeads: Int?
    public let outputProjDims: Int?
    
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case outputProjDims = "output_proj_dims"
    }
}

public struct Gemma4Configuration: Codable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let kvHeads: Int
    public let ropeTheta: Float
    public let ropeLocalBaseFreq: Float
    public let ropeTraditional: Bool
    public let queryPreAttnScalar: Float?
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let maxPositionEmbeddings: Int
    public let ropeScaling: [String: StringOrNumber]?
    public let globalHeadDim: Int
    public let numKvSharedLayers: Int
    public let useDoubleWideMlp: Bool
    
    // MoE / Global KV Configurations
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?
    public let numGlobalKeyValueHeads: Int
    public let tieWordEmbeddings: Bool
    public let enableMoeBlock: Bool

    /// Fraction of global head_dim used for RoPE (default 0.25 for Gemma 4 global attn)
    public let globalRopePartialFactor: Float
    /// Final logit softcapping value (0 = disabled). Gemma 4 uses 30.0.
    public let finalLogitSoftcapping: Float
    /// Per-layer conditioning dimension (0 = disabled)
    public let hiddenSizePerLayerInput: Int
    /// Vocabulary size for per-layer embedding table (0 = disabled)
    public let vocabSizePerLayerInput: Int
    /// Attention logit softcapping value (typically 50.0).
    public let attentionLogitSoftcapping: Float
    
    public let visionConfig: Gemma4VisionConfigurationProxy?
    public let audioConfig: Gemma4AudioConfigurationProxy?

    public init(
        modelType: String, hiddenSize: Int, hiddenLayers: Int, intermediateSize: Int,
        attentionHeads: Int, headDim: Int, rmsNormEps: Float, vocabularySize: Int, kvHeads: Int,
        ropeTheta: Float, ropeLocalBaseFreq: Float, ropeTraditional: Bool,
        queryPreAttnScalar: Float?, slidingWindow: Int, slidingWindowPattern: Int,
        maxPositionEmbeddings: Int, ropeScaling: [String: StringOrNumber]? = nil,
        globalHeadDim: Int = 512, numKvSharedLayers: Int = 0, useDoubleWideMlp: Bool = false,
        tieWordEmbeddings: Bool = true,
        numExperts: Int? = nil, topKExperts: Int? = nil, moeIntermediateSize: Int? = nil,
        numGlobalKeyValueHeads: Int? = nil,
        hiddenSizePerLayerInput: Int = 0, vocabSizePerLayerInput: Int = 0,
        globalRopePartialFactor: Float = 0.25,
        finalLogitSoftcapping: Float = 0.0,
        attentionLogitSoftcapping: Float = 50.0,
        enableMoeBlock: Bool? = nil,
        visionConfig: Gemma4VisionConfigurationProxy? = nil,
        audioConfig: Gemma4AudioConfigurationProxy? = nil
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDim = headDim
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads
        self.ropeTheta = ropeTheta
        self.ropeLocalBaseFreq = ropeLocalBaseFreq
        self.ropeTraditional = ropeTraditional
        self.queryPreAttnScalar = queryPreAttnScalar
        self.slidingWindow = slidingWindow
        self.slidingWindowPattern = slidingWindowPattern
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeScaling = ropeScaling
        self.globalHeadDim = globalHeadDim
        self.numKvSharedLayers = numKvSharedLayers
        self.useDoubleWideMlp = useDoubleWideMlp
        self.tieWordEmbeddings = tieWordEmbeddings
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.numGlobalKeyValueHeads = numGlobalKeyValueHeads ?? kvHeads
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.globalRopePartialFactor = globalRopePartialFactor
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.attentionLogitSoftcapping = attentionLogitSoftcapping
        self.enableMoeBlock = enableMoeBlock ?? (numExperts != nil && numExperts! > 0)
        self.visionConfig = visionConfig
        self.audioConfig = audioConfig
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case ropeLocalBaseFreq = "rope_local_base_freq"

        // MoE
        case numExperts = "num_experts"
        case topKExperts = "num_experts_per_tok"
        case tieWordEmbeddings = "tie_word_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case enableMoeBlock = "enable_moe_block"
        
        // Per-layer cond
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        // Logit softcapping
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attentionLogitSoftcapping = "attention_logit_softcapping"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
    }

    // Top-level keys (outside text_config)
    enum TopLevelCodingKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case enableMoeBlock = "enable_moe_block"
    }

    public init(from decoder: Decoder) throws {
        let nestedContainer = try decoder.container(keyedBy: VLMCodingKeys.self)

        let container =
            if nestedContainer.contains(.textConfig) {
                try nestedContainer.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        modelType = try container.decode(String.self, forKey: .modelType)
        
        let tieWordOpt = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
        tieWordEmbeddings = tieWordOpt ?? true
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try container.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)

        let enableMoeOpt = try container.decodeIfPresent(Bool.self, forKey: .enableMoeBlock)
        enableMoeBlock = enableMoeOpt ?? (self.numExperts != nil && self.numExperts! > 0)

        numGlobalKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads) ?? (try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 42

        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10_000.0
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar)
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        attentionLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .attentionLogitSoftcapping) ?? 0.0
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? (hiddenLayers == 35 ? 5 : 6)
        maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        ropeScaling = try container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        globalHeadDim = try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        numKvSharedLayers = try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers) ?? 0
        useDoubleWideMlp = try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? false
        // Per-layer conditioning
        self.hiddenSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 0
        self.vocabSizePerLayerInput = try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput) ?? 0
        
        let rootContainer = try decoder.container(keyedBy: CodingKeys.self)
        self.visionConfig = try rootContainer.decodeIfPresent(Gemma4VisionConfigurationProxy.self, forKey: .visionConfig)
        self.audioConfig = try rootContainer.decodeIfPresent(Gemma4AudioConfigurationProxy.self, forKey: .audioConfig)

        // Parse partial_rotary_factor for global attention from rope_parameters.full_attention
        // Gemma 4 uses only 25% of global_head_dim (512) for positional encoding = 128 rotated dims.
        struct AC: CodingKey {
            var stringValue: String; init?(stringValue s: String) { stringValue = s }
            var intValue: Int? { nil }; init?(intValue _: Int) { nil }
        }
        if let nestedContainer = try? decoder.container(keyedBy: AC.self),
           let ropeParamsContainer = try? nestedContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "rope_parameters")!),
           let fullAttnContainer = try? ropeParamsContainer.nestedContainer(keyedBy: AC.self, forKey: AC(stringValue: "full_attention")!),
           let prf = try? fullAttnContainer.decode(Float.self, forKey: AC(stringValue: "partial_rotary_factor")!) {
            self.globalRopePartialFactor = prf
        } else {
            self.globalRopePartialFactor = 0.25  // Gemma 4 default: 128/512
        }
    }
}

public class Gemma4Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let repeats: Int
    let headDim: Int
    let layerIdx: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int
    let slidingWindowPattern: Int
    let eps: Float
    let globalRopePartialFactor: Float
    /// QK attention logit softcapping (Gemma 4 uses 30.0). 0 = disabled.
    let attnLogitSoftcap: Float

    @ModuleInfo(key: "q_proj") public var queryProj: Linear
    @ModuleInfo(key: "k_proj") public var keyProj: Linear
    @ModuleInfo(key: "v_proj") public var valueProj: Linear
    @ModuleInfo(key: "o_proj") public var outputProj: Linear

    @ModuleInfo(key: "q_norm") public var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") public var keyNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") public var valueNorm: Gemma4RMSNormNoScale?

    @ModuleInfo public var rope: OffsetLayer

    public let numKvSharedLayers: Int
    public let hiddenLayers: Int

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.numKvSharedLayers = config.numKvSharedLayers
        self.hiddenLayers = config.hiddenLayers
        let dim = config.hiddenSize
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0
        self.eps = config.rmsNormEps
        
        self.nHeads = config.attentionHeads
        self.nKVHeads = self.isSliding ? config.kvHeads : config.numGlobalKeyValueHeads
        self.repeats = nHeads / (nKVHeads > 0 ? nKVHeads : 1)
        self.headDim = self.isSliding ? config.headDim : config.globalHeadDim

        // vLLM gemma4.py line 295: self.scaling = 1.0
        // "Gemma4 uses scaling=1.0. Q/K norms with learnable weights handle scaling implicitly."
        self.scale = 1.0
        self.attnLogitSoftcap = config.attentionLogitSoftcapping

        self._queryProj.wrappedValue = Linear(dim, nHeads * self.headDim, bias: false)
        self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * self.headDim, dim, bias: false)

        self._queryNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._keyNorm.wrappedValue = Gemma4RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)

        let ropeFactor = config.globalRopePartialFactor
        self.globalRopePartialFactor = ropeFactor

        if isSliding {
            // Sliding attention: standard RoPE on full head_dim
            self.rope = RoPE(
                dimensions: headDim, traditional: false,
                base: config.ropeLocalBaseFreq, scale: 1.0)
        } else {
            // Global attention: Proportional RoPE with partial_rotary_factor from config
            // vLLM Gemma4RotaryEmbedding: head_size=512, rotary_dim=0.25*512=128
            self.rope = Gemma4ProportionalRoPE(
                dims: self.headDim,
                traditional: false,
                theta: config.ropeTheta,
                partialRotaryFactor: ropeFactor  // 0.25 from config
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        let isKvSharedLayer = self.numKvSharedLayers > 0 && layerIdx >= (self.hiddenLayers - self.numKvSharedLayers)

        // Q projection (always runs, even for KV-shared layers)
        // vLLM gemma4.py lines 404-410, llama.cpp gemma4-iswa.cpp lines 64-76
        var queries = queryProj(x)
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        queries = queryNorm(queries)

        let LCache: Int
        if isKvSharedLayer {
            // The shared cache was already updated by the concrete layer, so its offset includes L.
            LCache = max(0, (cache?.offset ?? 0) - L)
        } else {
            // Concrete layers update their cache *after* applying RoPE.
            LCache = cache?.offset ?? 0
        }
        let output: MLXArray

        if isKvSharedLayer {
            // KV-shared layer: skip K/V projection, norm, RoPE, and cache update.
            // Only apply RoPE to Q. Read K/V from the designated shared cache.
            // vLLM gemma4.py lines 422-424, llama.cpp gemma4-iswa.cpp lines 105-109
            queries = rope(queries, offset: LCache)

            // Read K/V directly from the cache (populated by the concrete layer)
            let cachedKeys = (cache as? KVCacheSimple)?.keys
            let cachedValues = (cache as? KVCacheSimple)?.values
            if let cachedKeys, let cachedValues {
                let offset = cache?.offset ?? 0
                let slicedKeys = cachedKeys[0..., 0..., 0 ..< offset, 0...]
                let slicedValues = cachedValues[0..., 0..., 0 ..< offset, 0...]
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: slicedKeys, values: slicedValues,
                    scale: scale, mask: mask
                )
            } else {
                // Fallback: cache not yet populated, use self-projected K/V
                var keys = keyProj(x)
                var values = valueProj(x)
                keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
                values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
                keys = keyNorm(keys)
                if let vn = valueNorm { values = vn(values) }
                keys = rope(keys, offset: LCache)
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values,
                    scale: scale, mask: mask
                )
            }
        } else {
            // Non-shared layer: full K/V projection, norm, RoPE, cache update
            // vLLM gemma4.py lines 412-417, llama.cpp gemma4-iswa.cpp lines 79-100
            var keys = keyProj(x)
            var values = valueProj(x)

            keys = keys.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)

            keys = keyNorm(keys)
            if let vn = valueNorm {
                values = vn(values)
            }

            keys = rope(keys, offset: LCache)
            queries = rope(queries, offset: LCache)

            if attnLogitSoftcap > 0 {
                queries = queries * scale
                let (cachedKeys, cachedValues) = cache?.update(keys: keys, values: values) ?? (keys, values)
                var fullKeys = cachedKeys
                var fullValues = cachedValues
                // TurboKV decode if needed
                if let kvCache = cache as? KVCacheSimple,
                   let pk = kvCache.polarKeys, let pv = kvCache.polarValues,
                   kvCache.compressedOffset > 0 {
                    var histK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
                    var histV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
                    if kvCache.turboSplitHeads {
                        let B = histK.dim(0), H2 = histK.dim(1), T = histK.dim(2)
                        histK = histK.reshaped(B, H2 / 2, T, 512)
                        histV = histV.reshaped(B, H2 / 2, T, 512)
                    }
                    fullKeys   = MLX.concatenated([histK, cachedKeys],   axis: 2)
                    fullValues = MLX.concatenated([histV, cachedValues],   axis: 2)
                }
                var k = fullKeys
                var v = fullValues
                if nHeads > nKVHeads {
                    k = MLX.repeated(k, count: repeats, axis: 1)
                    v = MLX.repeated(v, count: repeats, axis: 1)
                }
                let scores = queries.matmul(k.transposed(0, 1, 3, 2))
                let maskArray: MLXArray?
                switch mask {
                case .none:
                    maskArray = nil
                case .causal:
                    maskArray = createCausalMask(n: L, offset: LCache, windowSize: isSliding ? slidingWindow : nil)
                case .array(let m):
                    maskArray = m
                case .arrays(let a):
                    maskArray = a.first
                @unknown default:
                    maskArray = nil
                }
                let originalType = scores.dtype
                var scoresF32 = scores.asType(.float32)
                let cap = MLXArray(attnLogitSoftcap).asType(.float32)
                scoresF32 = (MLX.tanh(scoresF32 / cap) * cap)
                if let maskArray {
                    scoresF32 = scoresF32 + maskArray.asType(.float32)
                }
                let attnWeights = MLX.softmax(scoresF32, axis: -1, precise: true).asType(originalType)
                output = matmul(attnWeights, v)
            } else {
                output = attentionWithCacheUpdate(
                    queries: queries, keys: keys, values: values,
                    cache: cache, scale: scale, mask: mask)
            }
        }
        return outputProj(
            output.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        )
    }
}

public class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Python reference: nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x)
        return downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

class Gemma4Router: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo var scale: MLXArray
    @ModuleInfo var per_expert_scale: MLXArray

    let eps: Float
    let scalarRootSize: Float

    public init(dimensions: Int, numExperts: Int, eps: Float) {
        self.eps = eps
        self.scalarRootSize = 1.0 / sqrt(Float(dimensions))
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([dimensions])
        self._per_expert_scale.wrappedValue = MLXArray.ones([numExperts])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, topK: Int) -> (MLXArray, MLXArray) {
        // Python reference: RMSNormNoScale → scale multiply → 1/sqrt(hidden) → proj → softmax → topK
        // RMS norm without learnable weight (None weight)
        let xF32 = x.asType(.float32)
        let meanX2 = MLX.mean(MLX.square(xF32), axes: [-1], keepDims: true)
        let xNormed = (x * MLX.rsqrt(meanX2 + MLXArray(eps))).asType(x.dtype)

        // Scale: x * root_size * scale (Python: x * self._root_size * self.scale)
        let scaled = xNormed * MLXArray(scalarRootSize) * scale

        let expertScores = proj(scaled)
        let routerProbs = MLX.softmax(expertScores, axis: -1)

        // Python: argpartition(-expert_scores, kth=topK-1)[..., :topK]
        let negScores = MLX.negative(expertScores)
        let allInds = MLX.argPartition(negScores, kth: topK - 1, axis: -1)
        let topKInds = allInds[0..., 0..., 0..<topK]

        // Gather the softmax probs for the selected experts
        var topKWeights = MLX.takeAlong(routerProbs, topKInds, axis: -1)

        // L1 normalize then apply per-expert scale
        topKWeights = topKWeights / topKWeights.sum(axis: -1, keepDims: true)
        topKWeights = topKWeights * MLX.takeAlong(per_expert_scale, topKInds, axis: -1)

        return (topKWeights, topKInds)
    }
}

class Gemma4SparseMoeBlock: Module {
    let topK: Int

    @ModuleInfo(key: "switch_glu") var switchGLU: SwitchGLU
    @ModuleInfo(key: "router") var router: Gemma4Router

    init(dimensions: Int, numExperts: Int, topK: Int, moeIntermediateSize: Int) {
        self.topK = topK
        self._router.wrappedValue = Gemma4Router(dimensions: dimensions, numExperts: numExperts, eps: 1e-6)
        self._switchGLU.wrappedValue = SwitchGLU(
            inputDims: dimensions,
            hiddenDims: moeIntermediateSize,
            numExperts: numExperts,
            activation: geluApproximate
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, routerInput: MLXArray) -> MLXArray {
        let (scores, inds) = router(routerInput, topK: topK)
        let y = switchGLU(x, inds)

        let B = y.dim(0)
        let T = y.dim(1)
        
        let yMasked = y * scores[0..., 0..., 0..., .newAxis]
        let yMerged = yMasked.sum(axis: 2)
        
        return yMerged.reshaped([B, T, -1])
    }
}

public class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") public var attention: Gemma4Attention
    @ModuleInfo public var mlp: Gemma4MLP
    @ModuleInfo(key: "experts") var expertsBlock: Gemma4SparseMoeBlock?

    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") public var preFFWNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") public var postFFWNorm: Gemma4RMSNorm

    // Per-layer conditioning (Gemma 4 architectural novelty)
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear?
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    @ModuleInfo(key: "layer_scalar") public var layerScalar: MLXArray

    let config: Gemma4Configuration
    let layerIdx: Int

    public init(_ config: Gemma4Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx

        self._attention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)

        self._inputLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFFWNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFFWNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            // per_layer_projection: [hiddenSizePerLayerInput → hiddenSize]
            self._perLayerProjection.wrappedValue = Linear(config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            // per_layer_input_gate: [hiddenSize → hiddenSizePerLayerInput]
            self._perLayerInputGate.wrappedValue = Linear(config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil, cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Gemma 4 decoder layer forward pass
        // Reference: vLLM gemma4.py Gemma4DecoderLayer.forward() lines 586-646
        //            llama.cpp gemma4-iswa.cpp lines 40-237
        //
        // Order: input_norm → attn → post_attn_norm → ADD residual
        //        → pre_ff_norm → mlp → post_ff_norm → ADD residual
        //        → PLE (gate, gelu, mul, project, norm, add)
        //        → layer_scalar

        var residual = x

        // 1. Attention
        var h = inputLayerNorm(residual)
        h = attention(h, mask: mask ?? .none, cache: cache)
        h = postAttentionNorm(h)
        var out = residual + h

        // 2. Feedforward
        residual = out
        h = preFFWNorm(residual)
        h = mlp(h)
        h = postFFWNorm(h)
        out = residual + h

        // 3. PLE (Per-Layer Embedding) — applied AFTER feedforward residual
        // vLLM gemma4.py lines 632-640, llama.cpp gemma4-iswa.cpp lines 203-223
        if let perLayerInput = perLayerInput,
           let perLayerGate = perLayerInputGate,
           let perLayerProj = perLayerProjection,
           let postPerLayerNorm = postPerLayerInputNorm {
            let gate = geluApproximate(perLayerGate(out))
            let gatedInput = gate * perLayerInput
            let conditioned = perLayerProj(gatedInput)
            let normed = postPerLayerNorm(conditioned)
            out = out + normed
        }

        // 4. Layer scalar — applied to final output
        // vLLM gemma4.py line 644, llama.cpp gemma4-iswa.cpp lines 227-229
        out = out * layerScalar

        return out
    }
}


public class Gemma4ModelInner: Module, LayerPartitionable {

    @ModuleInfo(key: "embed_tokens") public var embedTokens: QuantizedEmbedding
    @ModuleInfo public var layers: [Gemma4TransformerBlock]
    @ModuleInfo var norm: Gemma4RMSNorm

    // Per-layer conditioning weights (Gemma 4 architectural novelty)
    @ModuleInfo(key: "embed_tokens_per_layer") public var embedTokensPerLayer: QuantizedEmbedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: Gemma4RMSNorm?

    public let config: Gemma4Configuration

    // LayerPartitionable
    public var gpuLayerCount: Int? = nil
    public var totalLayerCount: Int { layers.count }

    init(_ config: Gemma4Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = QuantizedEmbedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
            groupSize: 64,
            bits: 4
        )

        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            // embed_tokens_per_layer yields exactly hiddenSizePerLayerInput dimensions per layer 
            self._embedTokensPerLayer.wrappedValue = QuantizedEmbedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput,
                groupSize: 64,
                bits: 4
            )
            // per_layer_model_projection: [hiddenSize → numLayers × hiddenSizePerLayerInput]
            self._perLayerModelProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenLayers * config.hiddenSizePerLayerInput,
                bias: false
            )
            self._perLayerProjectionNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSizePerLayerInput, eps: config.rmsNormEps)
        }

        super.init()
    }

    public func callAsFunction(
        _ inputs: MLXArray, inputEmbedding: MLXArray? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding = inputEmbedding {
            h = inputEmbedding
        } else {
            h = embedTokens(inputs)
            h = h * MLXArray(Float(config.hiddenSize).squareRoot())
        }
        var layerCache = cache
        if layerCache == nil {
            layerCache = Array(repeating: nil as KVCache?, count: layers.count)
        }

        // Use the correct concrete cache layer for mask creation
        // For shared KV models, the last cache entry (shared layer) is never written to.
        // Use the last concrete global and sliding layers instead.
        let firstSharedIdx = config.numKvSharedLayers > 0 ? config.hiddenLayers - config.numKvSharedLayers : config.hiddenLayers
        
        // Find first concrete global layer (for global mask)
        var globalCacheIdx = config.slidingWindowPattern - 1  // First global layer
        for i in stride(from: firstSharedIdx - 1, through: 0, by: -1) {
            if i % config.slidingWindowPattern == config.slidingWindowPattern - 1 {
                globalCacheIdx = i
                break
            }
        }
        // Find first concrete sliding layer (for sliding mask)
        var slidingCacheIdx = 0
        for i in 0 ..< firstSharedIdx {
            if i % config.slidingWindowPattern != config.slidingWindowPattern - 1 {
                slidingCacheIdx = i
                break
            }
        }
        
        let globalMask = createAttentionMask(h: h, cache: cache?[globalCacheIdx])
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode = 
            config.slidingWindowPattern > 1 
            ? createAttentionMask(h: h, cache: cache?[slidingCacheIdx], windowSize: config.slidingWindow)
            : .none

        // Compute per-layer conditioning tensor: [B, L, numLayers, hiddenSizePerLayerInput]
        var perLayerInputs: MLXArray? = nil
        if config.hiddenSizePerLayerInput > 0,
           let embedPerLayer = embedTokensPerLayer,
           let modelProj = perLayerModelProjection,
           let projNorm = perLayerProjectionNorm
        {
            let B = inputs.dim(0)
            let L = inputs.dim(1)
            let nL = config.hiddenLayers
            let D = config.hiddenSizePerLayerInput

            // Token-based per-layer embeddings
            // vLLM: per_layer_embeds * embed_scale_per_layer (= sqrt(hidden_size_per_layer_input))
            // llama.cpp: inp_per_layer = ggml_scale(inp_per_layer, sqrtf(n_embd_per_layer))
            let embedScale = MLXArray(Float(D).squareRoot()).asType(h.dtype)
            var tokenEmbeds = (embedPerLayer(inputs) * embedScale)
                .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

            // MASK OUT MULTIMODAL TOKENS: Zero out text-space embeddings for visual/audio tokens
            let isTextToken = MLX.logicalOr(MLX.less(inputs, MLXArray(258880)), MLX.greater(inputs, MLXArray(258884)))
            let expandedMask = isTextToken.reshaped([B, L, 1, 1]).asType(tokenEmbeds.dtype)
            tokenEmbeds = tokenEmbeds * expandedMask

            let projScale = MLXArray(1.0 / sqrt(Float(config.hiddenSize))).asType(h.dtype)
            let modelProjected = (modelProj(h) * projScale)
                .reshaped(B, L, nL, D)
            let modelProjectedNormed = projNorm(modelProjected)

            // Combine
            let combineScale = MLXArray(Float(1.0 / 2.0.squareRoot())).asType(h.dtype)
            perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
        }

        // sharedFullIdx and sharedSlidingIdx use firstSharedIdx from above
        // Find the last global and sliding layers before the shared block
        var sharedFullIdx = firstSharedIdx - 1
        while sharedFullIdx >= 0 && (sharedFullIdx % config.slidingWindowPattern != config.slidingWindowPattern - 1) {
            sharedFullIdx -= 1
        }
        var sharedSlidingIdx = firstSharedIdx - 1
        while sharedSlidingIdx >= 0 && (sharedSlidingIdx % config.slidingWindowPattern == config.slidingWindowPattern - 1) {
            sharedSlidingIdx -= 1
        }

        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? globalMask : slidingWindowMask
            
            let actualCacheIdx: Int
            if config.numKvSharedLayers > 0 && i >= firstSharedIdx {
                actualCacheIdx = isGlobal ? sharedFullIdx : sharedSlidingIdx
            } else {
                actualCacheIdx = i
            }
            let layerCache = cache?[actualCacheIdx]

            h = layer(h, mask: layerMask, cache: layerCache, perLayerInput: perLayerInputs?[0..., 0..., i, 0...])
        }

        return norm(h)
    }
}

public class Gemma4ModelInternal: Module, LayerPartitionable, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") public var model: Gemma4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { (0..<config.hiddenLayers).map { _ in config.kvHeads } }

    public var gpuLayerCount: Int? {
        get { model.gpuLayerCount }
        set { model.gpuLayerCount = newValue }
    }
    public var totalLayerCount: Int { model.totalLayerCount }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self._model.wrappedValue = Gemma4ModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
         return callAsFunction(inputs, inputEmbedding: nil, cache: cache)
    }

    public func callAsFunction(
        _ inputs: MLXArray, inputEmbedding: MLXArray? = nil, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var out = model(inputs, inputEmbedding: inputEmbedding, mask: mask, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        if config.finalLogitSoftcapping > 0 {
            var outF32 = out.asType(.float32)
            let cap = MLXArray(config.finalLogitSoftcapping).asType(.float32)
            outF32 = MLX.tanh(outF32 / cap) * cap
            out = outF32.asType(out.dtype)
        }
        return out
    }
    
    public func newCache(parameters: GenerateParameters? = nil) -> [KVCache] {
        var caches = [KVCache]()
        for i in 0 ..< config.hiddenLayers {
            if i % config.slidingWindowPattern == config.slidingWindowPattern - 1 {
                let cache = StandardKVCache()
                caches.append(cache)
            } else {
                caches.append(RotatingKVCache(maxSize: config.slidingWindow, keep: 0))
            }
        }
        return caches
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        var finalWeights = [String: MLXArray]()
        
        for (k, v) in weights {
            var newK = k
            
            // 1. Strip top-level VLM/Omni prefixes and filter out non-language-model keys
            if newK.hasPrefix("language_model.") {
                newK = String(newK.dropFirst("language_model.".count))
            } else if !newK.hasPrefix("model.") && !newK.hasPrefix("lm_head.") {
                // Ignore multimodal tower keys (vision_tower, audio_tower, etc)
                continue
            }
            
            // 2. Filter out unnecessary rotary or quantization metadata
            if newK.contains("self_attn.rotary_emb") || newK.contains("input_max") || newK.contains("input_min") || newK.contains("output_max") || newK.contains("output_min") {
                continue
            }
            
            // 3. Handle root-level PLE naming divergence
            if newK.hasPrefix("model.per_layer_projection.") {
                newK = newK.replacingOccurrences(of: "model.per_layer_projection", with: "model.per_layer_model_projection")
            }

            // 4. Experts gate splitting (Gemma 4 MoE)
            if newK.hasSuffix(".experts.gate_up_proj.weight") {
                  let base = newK.replacingOccurrences(of: ".experts.gate_up_proj.weight", with: ".experts.switch_glu")
                  let parts = MLX.split(v, parts: 2, axis: -2)
                  finalWeights["\(base).gate_proj.weight"] = parts[0]
                  finalWeights["\(base).up_proj.weight"] = parts[1]
                  continue
            }
            if newK.hasSuffix(".experts.down_proj.weight") {
                  let base = newK.replacingOccurrences(of: ".experts.down_proj.weight", with: ".experts.switch_glu.down_proj.weight")
                  finalWeights[base] = v
                  continue
            }
            
            // 5. Expert router mapping
            // Many HF exports use .router.weight, map into Swift's experts.router.proj structure
            newK = newK.replacingOccurrences(of: ".router.weight", with: ".experts.router.proj.weight")
            // Also handle any other .router keys (like scale, bias if they somehow exist)
            newK = newK.replacingOccurrences(of: ".router.", with: ".experts.router.")
            
            finalWeights[newK] = v
        }
        
        // Gemma 4 shares k_proj weights with v_proj in some layers (or all)
        for i in 0..<config.hiddenLayers {
            let kWeightKey = "model.layers.\(i).self_attn.k_proj.weight"
            let vWeightKey = "model.layers.\(i).self_attn.v_proj.weight"
            if finalWeights[kWeightKey] != nil && finalWeights[vWeightKey] == nil {
                finalWeights[vWeightKey] = finalWeights[kWeightKey]
            }
        }
        
        return finalWeights
    }
}

extension Gemma4ModelInternal: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0 as Module }
    }
}
