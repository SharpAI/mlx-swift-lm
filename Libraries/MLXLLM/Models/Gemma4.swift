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

public struct Gemma4AudioConfig: Codable {
    public let modelType: String
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let outputProjDims: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case outputProjDims = "output_proj_dims"
    }
}

public struct Gemma4VisionConfigProxy: Codable {
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let attentionHeads: Int?
    public let patchSize: Int?
    public let hiddenLayers: Int?
    
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case patchSize = "patch_size"
        case hiddenLayers = "num_hidden_layers"
    }
}

public struct Gemma4Configuration: Codable {
    public let audioConfig: Gemma4AudioConfig?
    public let visionConfig: Gemma4VisionConfigProxy?
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
    public let ropeGlobalBaseFreq: Float
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
    public let globalRopePartialFactor: Float?
    /// Final logit softcapping value (0 = disabled). Gemma 4 uses 30.0.
    public let finalLogitSoftcapping: Float
    public let attnLogitSoftcap: Float
    /// Per-layer conditioning dimension (0 = disabled)
    public let hiddenSizePerLayerInput: Int
    /// Vocabulary size for per-layer embedding table (0 = disabled)
    public let vocabSizePerLayerInput: Int

    public init(
        modelType: String,
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int,
        ropeTheta: Float,
        ropeLocalBaseFreq: Float,
        ropeGlobalBaseFreq: Float,
        ropeTraditional: Bool,
        queryPreAttnScalar: Float?,
        slidingWindow: Int,
        slidingWindowPattern: Int,
        maxPositionEmbeddings: Int,
        ropeScaling: [String: StringOrNumber]?,
        globalHeadDim: Int,
        numKvSharedLayers: Int,
        useDoubleWideMlp: Bool,
        tieWordEmbeddings: Bool,
        numExperts: Int? = nil,
        topKExperts: Int? = nil,
        moeIntermediateSize: Int? = nil,
        numGlobalKeyValueHeads: Int? = nil,
        hiddenSizePerLayerInput: Int = 0,
        vocabSizePerLayerInput: Int = 0,
        globalRopePartialFactor: Float? = nil,
        finalLogitSoftcapping: Float = 0.0,
        attnLogitSoftcap: Float = 0.0,
        audioConfig: Gemma4AudioConfig? = nil,
        visionConfig: Gemma4VisionConfigProxy? = nil,
        enableMoeBlock: Bool? = nil
    ) {
        self.audioConfig = audioConfig
        self.visionConfig = visionConfig
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
        self.ropeGlobalBaseFreq = ropeGlobalBaseFreq
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
        self.attnLogitSoftcap = attnLogitSoftcap
        self.enableMoeBlock = enableMoeBlock ?? (numExperts != nil && numExperts! > 0)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeLocalBaseFreq = "rope_local_base_freq"
        case ropeGlobalBaseFreq = "rope_global_base_freq"
        case ropeTraditional = "rope_traditional"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case globalHeadDim = "global_head_dim"
        case numKvSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        // MoE
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case enableMoeBlock = "enable_moe_block"
        // Per-layer conditioning
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        // Logit softcapping
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attnLogitSoftcap = "attn_logit_softcapping"
    }

    // Top-level keys (outside text_config)
    enum TopLevelCodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case audioConfig = "audio_config"
        case visionConfig = "vision_config"
    }

    enum VLMCodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case audioConfig = "audio_config"
        case visionConfig = "vision_config"
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

        audioConfig = try nestedContainer.decodeIfPresent(Gemma4AudioConfig.self, forKey: .audioConfig) ??
                      (try? decoder.container(keyedBy: TopLevelCodingKeys.self).decodeIfPresent(Gemma4AudioConfig.self, forKey: .audioConfig))
                      
        visionConfig = try nestedContainer.decodeIfPresent(Gemma4VisionConfigProxy.self, forKey: .visionConfig) ??
                       (try? decoder.container(keyedBy: TopLevelCodingKeys.self).decodeIfPresent(Gemma4VisionConfigProxy.self, forKey: .visionConfig))

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
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26

        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6912
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.0e-6
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 1
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        self.ropeLocalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeLocalBaseFreq) ?? 10000.0
        self.ropeGlobalBaseFreq = try container.decodeIfPresent(Float.self, forKey: .ropeGlobalBaseFreq) ?? 1000000.0
        ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        queryPreAttnScalar = try container.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar)
        finalLogitSoftcapping = try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        attnLogitSoftcap = try container.decodeIfPresent(Float.self, forKey: .attnLogitSoftcap) ?? 0.0
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

public class Gemma4RMSNormNoScale: Module {
    public let eps: Float

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

/// Standard Gemma 4 RMSNorm with direct weight scaling.
/// Unlike Gemma 1/2/3 which use additive-delta weights (1.0 + weight),
/// Gemma 4 stores the actual scale values directly in the checkpoint.
/// HF reference: normed_output = self._norm(x) * self.weight
public class Gemma4RMSNorm: Module {
    public let weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return MLXFast.rmsNorm(x, weight: self.weight, eps: self.eps)
    }
}

/// Proportional RoPE for Gemma 4 full-attention layers.
///
/// Frequencies are computed relative to the full head dimension, and rotation is
/// applied to the first rotated_dims//2 elements of each half of the head, matching
/// the HF `rotate_half` convention used in the Python reference.
public class Gemma4ProportionalRoPE: Module, OffsetLayer {
    public let dims: Int
    public let traditional: Bool
    public let rotatedDims: Int
    /// Stored as private; NOT a model parameter - computed at init from base/dims.
    private var _computedFreqs: MLXArray?

    public init(dims: Int, traditional: Bool = false, base: Float = 10000.0, partialRotaryFactor: Float = 1.0) {
        self.dims = dims
        self.traditional = traditional

        let ropeAngles = Int(partialRotaryFactor * Float(dims) / 2.0)
        self.rotatedDims = 2 * ropeAngles

        if rotatedDims > 0 {
            let exponents = MLXArray(stride(from: Float(0), to: Float(rotatedDims), by: 2)) / Float(dims)
            self._computedFreqs = MLXArray(base) ** exponents
        } else {
            self._computedFreqs = nil
        }

        super.init()
        // Freeze so the module system ignores this class entirely for weight loading
        self.freeze()
    }

    public func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        guard rotatedDims > 0, let freqs = _computedFreqs else { return x }

        let head = x[0..., 0..., 0..., 0..<dims]
        let half = dims / 2

        let left = head[0..., 0..., 0..., 0..<half]
        let right = head[0..., 0..., 0..., half...]

        let rotHalf = rotatedDims / 2

        // Gather the rotated portions from each half
        let leftRot = left[0..., 0..., 0..., 0..<rotHalf]
        let rightRot = right[0..., 0..., 0..., 0..<rotHalf]
        let rotated = concatenated([leftRot, rightRot], axis: -1)

        // Apply standard RoPE with pre-computed freqs
        let rotatedResult = MLXFast.RoPE(
            rotated, dimensions: rotatedDims, traditional: traditional,
            base: nil, scale: 1.0, offset: offset, freqs: freqs)

        // Reconstruct: put rotated portions back into their halves
        let leftPassthru = left[0..., 0..., 0..., rotHalf...]
        let rightPassthru = right[0..., 0..., 0..., rotHalf...]

        let newLeft = concatenated([rotatedResult[0..., 0..., 0..., 0..<rotHalf], leftPassthru], axis: -1)
        let newRight = concatenated([rotatedResult[0..., 0..., 0..., rotHalf...], rightPassthru], axis: -1)

        return concatenated([newLeft, newRight], axis: -1)
    }
}

class Gemma4Attention: Module {
    public let nHeads: Int
    public let nKVHeads: Int
    public let repeats: Int
    public let headDim: Int
    public let layerIdx: Int
    public let scale: Float
    public let isSliding: Bool
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let eps: Float
    public let globalRopePartialFactor: Float
    /// QK attention logit softcapping (Gemma 4 uses 30.0). 0 = disabled.
    public let attnLogitSoftcap: Float
    public let queryPreAttnScalar: Float

    @ModuleInfo(key: "q_proj") var queryProj: Linear
    @ModuleInfo(key: "k_proj") var keyProj: Linear
    @ModuleInfo(key: "v_proj") var valueProj: Linear
    @ModuleInfo(key: "o_proj") var outputProj: Linear

    @ModuleInfo(key: "q_norm") var queryNorm: Gemma4RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: Gemma4RMSNorm
    @ModuleInfo(key: "v_norm") var valueNorm: Gemma4RMSNormNoScale

    @ModuleInfo var rope: OffsetLayer

    // === KV Cache Sharing State (Gemma 4 Novelty) ===
    public let isKVSharedLayer: Bool

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        let dim = config.hiddenSize
        self.layerIdx = layerIdx
        self.slidingWindow = config.slidingWindow
        self.slidingWindowPattern = config.slidingWindowPattern
        self.isSliding = (layerIdx + 1) % config.slidingWindowPattern != 0
        self.eps = config.rmsNormEps
        
        // KV Sharing configuration
        let prevLayersCount = config.hiddenLayers - config.numKvSharedLayers
        self.isKVSharedLayer = config.numKvSharedLayers > 0 && layerIdx >= prevLayersCount
        
        self.nHeads = config.attentionHeads
        self.nKVHeads = self.isSliding ? config.kvHeads : config.numGlobalKeyValueHeads
        self.repeats = nHeads / (nKVHeads > 0 ? nKVHeads : 1)
        self.headDim = self.isSliding ? config.headDim : config.globalHeadDim

        // Python reference: self.scale = 1.0 — Q/K RMS norms handle magnitude.
        self.scale = 1.0
        self.attnLogitSoftcap = config.attnLogitSoftcap
        self.queryPreAttnScalar = config.queryPreAttnScalar ?? 1.0

        self._queryProj.wrappedValue = Linear(dim, nHeads * self.headDim, bias: false)
        self._outputProj.wrappedValue = Linear(nHeads * self.headDim, dim, bias: false)
        self._queryNorm.wrappedValue = Gemma4RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)

        // Unused dummy modules for shared layers that still exist in the checkpoint
        if !self.isKVSharedLayer {
            self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
            self._keyNorm.wrappedValue = Gemma4RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
            // vNorm is purely unscaled RMS
            self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        } else {
            // Unused modules for shared layers to satisfy @ModuleInfo matching the checkpoint shapes
            self._keyProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
            self._valueProj.wrappedValue = Linear(dim, nKVHeads * self.headDim, bias: false)
            self._keyNorm.wrappedValue = Gemma4RMSNorm(dimensions: self.headDim, eps: config.rmsNormEps)
            self._valueNorm.wrappedValue = Gemma4RMSNormNoScale(eps: config.rmsNormEps)
        }

        let ropeFactor = config.globalRopePartialFactor ?? 0.25
        self.globalRopePartialFactor = ropeFactor

        if isSliding {
            // Sliding attention: standard RoPE on full head_dim
            self.rope = RoPE(
                dimensions: headDim, traditional: false,
                base: config.ropeLocalBaseFreq, scale: 1.0)
        } else {
            // Global attention: ProportionalRoPE with partial rotation
            self.rope = Gemma4ProportionalRoPE(
                dims: self.headDim,
                traditional: false,
                base: config.ropeGlobalBaseFreq,  // rope_parameters.full_attention.rope_theta
                partialRotaryFactor: ropeFactor
            )
        }

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        sharedKV: (MLXArray, MLXArray, Int)? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = queryProj(x)
        queries = queries.reshaped(B, L, nHeads, -1).transposed(0, 2, 1, 3)
        queries = queryNorm(queries)

        let LCache = sharedKV?.2 ?? (cache?.offset ?? 0)
        queries = rope(queries, offset: LCache)
        
        // Gemma 3/4 specific query pre-attention scalar (typically 256.0) required to 
        // prevent flat 0.0 distributions causing gibberish across normalized QK arrays
        if queryPreAttnScalar != 1.0 {
            queries = queries * MLXArray(queryPreAttnScalar).asType(queries.dtype)
        }

        var keys: MLXArray
        var values: MLXArray

        if isKVSharedLayer, let shared = sharedKV {
            keys = shared.0
            values = shared.1
        } else {
            keys = keyProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            values = valueProj(x).reshaped(B, L, nKVHeads, -1).transposed(0, 2, 1, 3)
            
            keys = keyNorm(keys)
            values = valueNorm(values)
            
            keys = rope(keys, offset: LCache)
        }

        let output: MLXArray
        if attnLogitSoftcap > 0 {
            // Gemma 4 uses QK attention logit softcapping before softmax:
            //   scores = tanh(scores / cap) * cap  (llama.cpp llm_build_gemma4_iswa)
            // MLXFast.scaledDotProductAttention has no softcap parameter, so we do it manually.
            let cachedKeys: MLXArray
            let cachedValues: MLXArray
            if isKVSharedLayer {
                // Shared layers already receive the full KV history directly.
                // Do NOT call cache.update because it would repeatedly append the full history!
                cachedKeys = keys
                cachedValues = values
            } else {
                let kvTuple = cache?.update(keys: keys, values: values) ?? (keys, values)
                cachedKeys = kvTuple.0
                cachedValues = kvTuple.1
            }
            var fullKeys = cachedKeys
            var fullValues = cachedValues
            // TurboKV decode if needed
            if let kvCache = cache as? KVCacheSimple,
               let pk = kvCache.polarKeys, let pv = kvCache.polarValues,
               kvCache.compressedOffset > 0 {
                var histK = MLXFast.turboDecodeK(packed: pk).asType(cachedKeys.dtype)
                var histV = MLXFast.turboDecodeV(packed: pv).asType(cachedValues.dtype)
                // Merge 2×256 virtual heads back to original count × 512
                if kvCache.turboSplitHeads {
                    let B = histK.dim(0), H2 = histK.dim(1), T = histK.dim(2)
                    histK = histK.reshaped(B, H2 / 2, T, 512)
                    histV = histV.reshaped(B, H2 / 2, T, 512)
                }
                fullKeys   = concatenated([histK, cachedKeys],   axis: 2)
                fullValues = concatenated([histV, cachedValues], axis: 2)
            }
            // GQA expansion
            var k = fullKeys
            var v = fullValues
            if nHeads > nKVHeads {
                k = MLX.repeated(k, count: repeats, axis: 1)
                v = MLX.repeated(v, count: repeats, axis: 1)
            }
            // scores: [B, nH, L, S]
            // We MUST upcast to float32 before the matmul!
            // queryPreAttnScalar (256.0) forces the Float16 dot product to easily exceed the 65504
            // capacity limit during accumulation, leading to Inf/NaN outputs and gibberish token generation.
            var scores = (queries.asType(.float32) * scale).matmul(k.asType(.float32).transposed(0, 1, 3, 2))

            // Apply QK softcap FIRST: tanh(scores / cap) * cap
            // Critically, we MUST evaluate tanh in float32 to avoid saturation.
            // Also critically, this MUST be before the mask, otherwise -1e9 mask gets clipped to -30.0!
            let cap = MLXArray(attnLogitSoftcap).asType(.float32)
            scores = (MLX.tanh(scores / cap) * cap)
            
            // Apply attention mask AFTER softcap
            if let maskArray = mask.mask {
                scores = scores + maskArray.asType(.float32)
            }
            
            // Softmax + weighted sum
            let attnWeights = MLX.softmax(scores, axis: -1).asType(v.dtype)
            output = matmul(attnWeights, v)
        } else {
            // Unsoftcapped attention logic (natively handles update internally)
            // If caching is meant to bypass (i.e. shared layer), pass nil
            let activeCache = isKVSharedLayer ? nil : cache
            output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values,
                cache: activeCache, scale: scale, mask: mask)
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
    @ModuleInfo(key: "scale") var scale: MLXArray
    @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    public let eps: Float
    public let scalarRootSize: Float

    public init(dimensions: Int, numExperts: Int, eps: Float) {
        self.eps = eps
        self.scalarRootSize = 1.0 / sqrt(Float(dimensions))
        self._proj.wrappedValue = Linear(dimensions, numExperts, bias: false)
        self._scale.wrappedValue = MLXArray.ones([dimensions])
        self._perExpertScale.wrappedValue = MLXArray.ones([numExperts])
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
        topKWeights = topKWeights * perExpertScale[topKInds]

        return (topKWeights, topKInds)
    }
}

class Gemma4SparseMoeBlock: Module {
    public let topK: Int

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

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "experts") var expertsBlock: Gemma4SparseMoeBlock?

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: Gemma4RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: Gemma4RMSNorm

    // MoE specific norms
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: Gemma4RMSNorm?
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: Gemma4RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: Gemma4RMSNorm?

    // Per-layer conditioning (Gemma 4 architectural novelty)
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var perLayerProjectionLayer: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: Gemma4RMSNorm?

    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    public let numAttentionHeads: Int
    public let hiddenSize: Int
    public let layerIdx: Int
    public let isMoe: Bool
    public let hasPerLayerInput: Bool

    init(_ config: Gemma4Configuration, layerIdx: Int) {
        self.numAttentionHeads = config.attentionHeads
        self.hiddenSize = config.hiddenSize
        self.layerIdx = layerIdx
        self.hasPerLayerInput = config.hiddenSizePerLayerInput > 0

        self._selfAttention.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)

        let mlpSize: Int
        if config.useDoubleWideMlp && layerIdx >= (config.hiddenLayers - config.numKvSharedLayers) {
            mlpSize = config.intermediateSize * 2
        } else {
            mlpSize = config.intermediateSize
        }
        self.mlp = Gemma4MLP(dimensions: config.hiddenSize, hiddenDimensions: mlpSize)

        self.isMoe = config.enableMoeBlock
        
        if self.isMoe {
            let numExperts = config.numExperts ?? 1
            self._expertsBlock.wrappedValue = Gemma4SparseMoeBlock(
                dimensions: config.hiddenSize,
                numExperts: numExperts,
                topK: config.topKExperts ?? 1,
                moeIntermediateSize: config.moeIntermediateSize ?? config.intermediateSize
            )
            self._postFeedforwardLayerNorm1.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._preFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            self._postFeedforwardLayerNorm2.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        if hasPerLayerInput {
            self._perLayerInputGate.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSizePerLayerInput, bias: false)
            self._perLayerProjectionLayer.wrappedValue = Linear(
                config.hiddenSizePerLayerInput, config.hiddenSize, bias: false)
            self._postPerLayerInputNorm.wrappedValue = Gemma4RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        self._inputLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayerNorm.wrappedValue = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache? = nil,
        perLayerInput: MLXArray? = nil,
        sharedKV: (MLXArray, MLXArray, Int)? = nil
    ) -> MLXArray {
        // === Stage 1: Self-Attention (Python reference pattern) ===
        // residual = hidden_states
        // hidden_states = input_layernorm(hidden_states)
        // hidden_states = self_attn(hidden_states)
        // hidden_states = post_attention_layernorm(hidden_states)
        // hidden_states = residual + hidden_states
        var h = x
        var residual = h
        h = selfAttention(inputLayerNorm(h), mask: mask, cache: cache, sharedKV: sharedKV)
        h = postAttentionLayerNorm(h)
        h = residual + h

        // === Stage 2: FFN / MoE (Python reference pattern) ===
        // residual = hidden_states
        // hidden_states = pre_feedforward_layernorm(hidden_states) → mlp → ...
        // hidden_states = post_feedforward_layernorm(hidden_states)
        // hidden_states = residual + hidden_states
        residual = h
        if isMoe {
            // Dense MLP path
            let denseOut = postFeedforwardLayerNorm1!(mlp(preFeedforwardLayerNorm(h)))

            // MoE sparse path (router takes pre-ffn residual = `residual`)
            let residualFlat = residual.reshaped(-1, residual.dim(-1))
            let sparsePreNorm = preFeedforwardLayerNorm2!(residualFlat)
            let sparseOut = expertsBlock!(sparsePreNorm, routerInput: residualFlat)
            let sparsePostNorm = postFeedforwardLayerNorm2!(
                sparseOut.reshaped(residual.shape))

            h = postFeedforwardLayerNorm(denseOut + sparsePostNorm)
        } else {
            h = postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(h)))
        }
        h = residual + h

        // === Stage 3: Per-Layer Input Conditioning (Gemma 4 novelty) ===
        // residual = hidden_states
        // hidden_states = gate(hidden_states)
        // hidden_states = act_fn(hidden_states) * per_layer_input
        // hidden_states = projection(hidden_states)
        // hidden_states = post_per_layer_input_norm(hidden_states)
        // hidden_states = residual + hidden_states
        if hasPerLayerInput,
           let pli = perLayerInput,
           let gate = perLayerInputGate,
           let proj = perLayerProjectionLayer,
           let plNorm = postPerLayerInputNorm
        {
            residual = h
            h = gate(h)             // Linear: hidden → hidden_size_per_layer_input
            h = geluApproximate(h)  // act_fn (gelu_approx)
            h = h * pli             // element-wise multiply by per_layer embedding
            h = proj(h)             // Linear: hidden_size_per_layer_input → hidden
            h = plNorm(h)           // post_per_layer_input_norm
            h = residual + h
        }

        // === Stage 4: Layer Scalar ===
        // hidden_states *= layer_scalar
        return h * layerScalar
    }
}

// Restored LayerPartitionable & StreamableMoE conformance to re-enable 
// SSD expert streaming, bridging the missing protocols from Damon Janis's initial draft.
// Reference: https://github.com/SharpAI/mlx-swift-lm/pull/1
public class Gemma4ModelInternal: Module, LayerPartitionable, StreamableMoE {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    fileprivate let layers: [Gemma4TransformerBlock]
    fileprivate let norm: Gemma4RMSNorm

    // Per-layer conditioning weights (Gemma 4 architectural novelty)
    @ModuleInfo(key: "embed_tokens_per_layer") public var embedTokensPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") public var perLayerModelProjection: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") public var perLayerProjectionNorm: Gemma4RMSNorm?

    public var loraLayers: [Module] { layers }

    public let config: Gemma4Configuration

    // LayerPartitionable
    public var gpuLayerCount: Int? = nil
    public var totalLayerCount: Int { layers.count }
    
    // StreamableMoE
    public var streamExperts: Bool = false

    public init(_ config: Gemma4Configuration) {
        self.config = config

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )

        self.layers = (0 ..< config.hiddenLayers).map { layerIdx in
            Gemma4TransformerBlock(config, layerIdx: layerIdx)
        }

        self.norm = Gemma4RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        if config.hiddenSizePerLayerInput > 0 {
            // embed_tokens_per_layer: [vocabSizePerLayerInput, numLayers × hiddenSizePerLayerInput]
            self._embedTokensPerLayer.wrappedValue = Embedding(
                embeddingCount: config.vocabSizePerLayerInput,
                dimensions: config.hiddenLayers * config.hiddenSizePerLayerInput
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
        _ inputs: MLXArray? = nil,
        inputEmbedding: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache?]? = nil
    ) -> MLXArray {
        var h: MLXArray
        if let inputEmbedding = inputEmbedding {
            h = inputEmbedding
        } else if let inputs = inputs {
            h = embedTokens(inputs)
            // Python reference: h = h * self.embed_scale where embed_scale = hidden_size**0.5
            h = h * MLXArray(Float(config.hiddenSize).squareRoot()).asType(h.dtype)
        } else {
            fatalError("Either inputs or inputEmbedding must be provided")
        }



        var layerCache = cache
        if layerCache == nil {
            if config.numKvSharedLayers > 0 {
                layerCache = (0..<layers.count).map { _ in KVCacheSimple() }
            } else {
                layerCache = Array(repeating: nil as KVCache?, count: layers.count)
            }
        }

        let globalMask = createAttentionMask(h: h, cache: cache?[config.slidingWindowPattern - 1])
        let slidingWindowMask: MLXFast.ScaledDotProductAttentionMaskMode =
            config.slidingWindowPattern > 1
            ? createAttentionMask(h: h, cache: cache?[0], windowSize: config.slidingWindow)
            : .none

        // DIAGNOSTIC: Re-enabling per-layer inputs for text generation stability.
        var perLayerInputs: MLXArray? = nil
        let _disablePerLayer = false  // Set to false to enable per-layer conditioning
        if !_disablePerLayer, config.hiddenSizePerLayerInput > 0,
           let embedPerLayer = embedTokensPerLayer,
           let modelProj = perLayerModelProjection,
           let projNorm = perLayerProjectionNorm
        {
            // If inputs is nil (e.g., using embeddings directly without tokens), skip per-layer inputs
            // or pass a zero tensor, or the user would need to supply it. 
            // In Omni pipelines, we usually supply tokens and inject embeddings post-hoc, 
            // but if inputs is nil, let's gracefully guard it.
            let src = inputs ?? h
            let B = src.ndim >= 2 ? src.dim(0) : 1
            let L = src.ndim >= 2 ? src.dim(1) : src.dim(0)
            let nL = config.hiddenLayers
            let D = config.hiddenSizePerLayerInput
            print("DEBUG: B=\(B), L=\(L), h.shape=\(h.shape)")

            if let inputs = inputs {
                // Token-based per-layer embeddings, scaled by sqrt(hiddenSizePerLayerInput)
                let tokenScale = MLXArray(sqrt(Float(D))).asType(h.dtype)
                let tokenEmbeds = (embedPerLayer(inputs) * tokenScale)
                    .reshaped(B, L, nL, D)  // [B, L, numLayers, D]

                // Model projection: scale by 1/sqrt(hidden_size) per reference
                // per_layer_model_projection_scale = hidden_size ** -0.5
                let modelProjScale = MLXArray(Float(1.0 / sqrt(Float(config.hiddenSize)))).asType(h.dtype)
                let modelProjected = (modelProj(h) * modelProjScale).reshaped(B, L, nL, D)
                let modelProjectedNormed = projNorm(modelProjected)

                // Combine: (token_embeds + projection) * 2^-0.5
                let combineScale = MLXArray(Float(1.0 / 2.0.squareRoot())).asType(h.dtype)
                perLayerInputs = (tokenEmbeds + modelProjectedNormed) * combineScale
            }
        }
        
        let prevLayersCount = config.hiddenLayers - config.numKvSharedLayers
        var lastGlobal = -1
        var lastSliding = -1
        for j in 0..<prevLayersCount {
            if j % config.slidingWindowPattern == config.slidingWindowPattern - 1 {
                lastGlobal = j
            } else {
                lastSliding = j
            }
        }
        
        for (i, layer) in layers.enumerated() {
            let isGlobal = (i % config.slidingWindowPattern == config.slidingWindowPattern - 1)
            let layerMask = isGlobal ? globalMask : slidingWindowMask
            // Slice per-layer conditioning for this layer: [B, L, D]
            let pli = perLayerInputs.map { $0[0..., 0..., i, 0...] }
            
            let sharedKV: (MLXArray, MLXArray, Int)?
            if let block = layer as? Gemma4TransformerBlock, block.selfAttention.isKVSharedLayer {
                let sourceIdx = isGlobal ? lastGlobal : lastSliding
                if sourceIdx >= 0, let sourceCache = layerCache?[sourceIdx], sourceCache.state.count >= 2 {
                    let k: MLXArray
                    let v: MLXArray
                    if let rotating = sourceCache as? RotatingKVCache, let keys = rotating.innerState().first, let values = rotating.innerState().last {
                        k = rotating.temporallyOrdered(keys)
                        v = rotating.temporallyOrdered(values)
                    } else {
                        let state = sourceCache.state
                        k = state[0]
                        v = state[1]
                    }
                    let ropeOffset = max(0, sourceCache.offset - h.dim(1))
                    sharedKV = (k, v, ropeOffset)
                } else {
                    sharedKV = nil
                }
            } else {
                sharedKV = nil
            }
            
            h = partitionedLayerCall(index: i, gpuLayerCount: gpuLayerCount, stream: streamExperts, cacheToEval: layerCache?[i]) {
                layer(h, mask: layerMask, cache: layerCache?[i], perLayerInput: pli, sharedKV: sharedKV)
            }

        }
        return norm(h)
    }
}

public class Gemma4Model: Module, LLMModel {

    @ModuleInfo public var model: Gemma4ModelInternal
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Gemma4Configuration
    public var vocabularySize: Int { config.vocabularySize }

    public init(_ config: Gemma4Configuration) {
        self.config = config
        self.model = Gemma4ModelInternal(config)
        // Always create a separate lm_head — following the Gemma 3 pattern.
        // For tied embeddings, sanitize() will copy embed_tokens weights to lm_head.
        // This ensures logit projection uses QuantizedLinear.quantizedMM rather than
        // QuantizedEmbedding.asLinear, which is critical for numerical accuracy.
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var out = model(inputs, mask: nil, cache: cache)
        out = lmHead!(out)

        // Apply final logit softcapping (Python reference line 579-580)
        if config.finalLogitSoftcapping > 0 {
            let originalType = out.dtype
            let outF32 = out.asType(.float32)
            let cap = MLXArray(config.finalLogitSoftcapping).asType(.float32)
            out = (MLX.tanh(outF32 / cap) * cap).asType(originalType)
        }
        return out
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        var processedWeights = weights

        let unflattened = ModuleParameters.unflattened(weights)
        if let lm = unflattened["language_model"] {
            processedWeights = Dictionary(uniqueKeysWithValues: lm.flattened())
        }

        let expectedVocab = config.vocabularySize
        let keysToCheck = [
            "model.embed_tokens.weight", "model.embed_tokens.scales", "model.embed_tokens.biases",
            "lm_head.weight", "lm_head.scales", "lm_head.biases",
        ]
        for key in keysToCheck {
            if let tensor = processedWeights[key], tensor.dim(0) > expectedVocab {
                processedWeights[key] = tensor[0 ..< expectedVocab]
            }
        }
        
        var finalWeights = [String: MLXArray]()
        for (k, v) in processedWeights {
            if k.contains("self_attn.rotary_emb") || k.contains("input_max") || k.contains("input_min") || k.contains("output_max") || k.contains("output_min") {
                continue
            }
            if k.hasSuffix(".experts.gate_up_proj.weight") {
                 let base = k.replacingOccurrences(of: ".experts.gate_up_proj.weight", with: ".experts.switch_glu")
                 let parts = MLX.split(v, parts: 2, axis: -2)
                 finalWeights["\(base).gate_proj.weight"] = parts[0]
                 finalWeights["\(base).up_proj.weight"] = parts[1]
                 continue
            }
            if k.hasSuffix(".experts.down_proj.weight") {
                 let base = k.replacingOccurrences(of: ".experts.down_proj.weight", with: ".experts.switch_glu.down_proj.weight")
                 finalWeights[base] = v
                 continue
            }
            let newK = k.replacingOccurrences(of: ".router.", with: ".experts.router.")
            finalWeights[newK] = v
        }

        // Explicitly map per_layer_projection_norm to ensure it survives flattening
        if let normWeight = weights["language_model.model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        } else if let normWeight = weights["model.per_layer_projection_norm.weight"] {
            finalWeights["model.per_layer_projection_norm.weight"] = normWeight
        }

        // Handle mixed-quantization: MLP and MoE experts might be 8-bit while other layers are 4-bit.
        for i in 0..<config.hiddenLayers {
            let wKey = "model.layers.\(i).experts.router.proj.weight"
            let sKey = "model.layers.\(i).experts.router.proj.scales"
            let bKey = "model.layers.\(i).experts.router.proj.biases"
            if let packedW = finalWeights[wKey],
               let scales = finalWeights[sKey],
               let biases = finalWeights[bKey] {
                let bits = 32 * packedW.shape.last! / (scales.shape.last! * 64)
                finalWeights[wKey] = MLX.dequantized(
                    packedW, scales: scales, biases: biases, groupSize: 64, bits: bits)
                finalWeights.removeValue(forKey: sKey)
                finalWeights.removeValue(forKey: bKey)
            }
        }

        // Per-layer conditioning weights are now fully implemented and loaded normally.
        
        // Gemma 4 shares k_proj weights with v_proj in some layers (or all)
        for i in 0..<config.hiddenLayers {
            let kWeightKey = "model.layers.\(i).self_attn.k_proj.weight"
            let vWeightKey = "model.layers.\(i).self_attn.v_proj.weight"
            if finalWeights[kWeightKey] != nil && finalWeights[vWeightKey] == nil {
                finalWeights[vWeightKey] = finalWeights[kWeightKey]
                
                let kScalesKey = "model.layers.\(i).self_attn.k_proj.scales"
                let vScalesKey = "model.layers.\(i).self_attn.v_proj.scales"
                if finalWeights[kScalesKey] != nil {
                    finalWeights[vScalesKey] = finalWeights[kScalesKey]
                }
                
                let kBiasesKey = "model.layers.\(i).self_attn.k_proj.biases"
                let vBiasesKey = "model.layers.\(i).self_attn.v_proj.biases"
                if finalWeights[kBiasesKey] != nil {
                    finalWeights[vBiasesKey] = finalWeights[kBiasesKey]
                }
            }
        }
        // For tied word embeddings, copy embed_tokens weights to lm_head.
        // This follows the Gemma 3 pattern — the load pipeline will auto-quantize
        // lm_head into QuantizedLinear, giving us numerically correct quantizedMM
        // for logit projection instead of the less precise Embedding.asLinear.
        if finalWeights["lm_head.weight"] == nil {
            ["weight", "scales", "biases"].forEach { key in
                if let embedWeight = finalWeights["model.embed_tokens.\(key)"] {
                    finalWeights["lm_head.\(key)"] = embedWeight
                }
            }
        }
        
        return finalWeights
    }

    public func newCache(parameters: GenerateParameters? = nil) -> [any KVCache] {
        var caches = [any KVCache]()
        let slidingWindow = config.slidingWindow
        let slidingWindowPattern = config.slidingWindowPattern

        for i in 0 ..< config.hiddenLayers {
            let isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1)

            if isGlobalLayer {
                let cache = StandardKVCache()
                cache.step = 1024
                caches.append(cache)
            } else {
                caches.append(
                    RotatingKVCache(maxSize: slidingWindow, keep: 0)
                )
            }
        }

        return caches
    }

}

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0 as Module }
    }
}
