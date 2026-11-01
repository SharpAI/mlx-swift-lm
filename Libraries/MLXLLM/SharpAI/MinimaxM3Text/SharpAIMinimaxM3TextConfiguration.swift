// SharpAI fork overlay: minimax_m3_text configuration. Not part of upstream mlx-swift-lm.
//
// Core keys use the HF names of the MiniMax config in MiniMax.swift. The M3 text keys below
// (dense/shared widths, routing scale, swiglu constants, layer schedules, sparse_attention_config)
// follow the text_config of MiniMaxAI/MiniMax-M3 config.json, read through a summary.
// UNVERIFIED: the flat top-level layout of a standalone minimax_m3_text config is not confirmed.
// The M3 config nests these keys under text_config, so this flat form is an assumption.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3TextError: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "MinimaxM3Text invalid configuration: \(detail)"
        case .unsupported(let detail):
            return "MinimaxM3Text unsupported: \(detail)"
        }
    }
}

/// The block-sparse index branch (`sparse_attention_config` in the M3 text config).
public struct SharpAIMinimaxM3TextSparseAttentionConfiguration: Decodable, Sendable {
    public let useSparseAttention: Bool
    /// Width of each indexer head (`sparse_index_dim`).
    public let indexDim: Int
    /// Indexer head count (`sparse_num_index_heads`). Query heads are grouped evenly over these.
    public let indexHeads: Int
    /// Key blocks kept per query and indexer head (`sparse_topk_blocks`).
    public let topkBlocks: Int
    /// Keys per block (`sparse_block_size`).
    public let blockSize: Int
    /// Blocks always kept for each query, starting at its own block (`sparse_local_block`).
    public let localBlocks: Int
    /// Per-layer 0/1 flags. Nil means layers at or after index 3 are sparse.
    public let attentionFreq: [Int]?
    /// Only `max` is implemented (`sparse_score_type`).
    public let scoreType: String

    enum CodingKeys: String, CodingKey {
        case useSparseAttention = "use_sparse_attention"
        case indexDim = "sparse_index_dim"
        case indexHeads = "sparse_num_index_heads"
        case topkBlocks = "sparse_topk_blocks"
        case blockSize = "sparse_block_size"
        case localBlocks = "sparse_local_block"
        case attentionFreq = "sparse_attention_freq"
        case scoreType = "sparse_score_type"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        useSparseAttention =
            try container.decodeIfPresent(Bool.self, forKey: .useSparseAttention) ?? true
        indexDim = try container.decodeIfPresent(Int.self, forKey: .indexDim) ?? 128
        indexHeads = try container.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 4
        topkBlocks = try container.decodeIfPresent(Int.self, forKey: .topkBlocks) ?? 16
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 128
        localBlocks = try container.decodeIfPresent(Int.self, forKey: .localBlocks) ?? 1
        attentionFreq = try container.decodeIfPresent([Int].self, forKey: .attentionFreq)
        scoreType = try container.decodeIfPresent(String.self, forKey: .scoreType) ?? "max"
    }
}

/// Top-level `config.json` for `model_type: minimax_m3_text`.
public struct SharpAIMinimaxM3TextConfiguration: Decodable, Sendable {
    public static let expectedModelType = "minimax_m3_text"
    public static let expectedArchitecture = "MiniMaxM3SparseForCausalLM"
    /// Layers before this index are dense MLP layers when `moe_layer_freq` is absent.
    static let defaultDenseLeadingLayers = 3

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int?
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let rotaryDim: Int?
    public let maxPositionEmbeddings: Int?
    public let numLocalExperts: Int?
    public let numExpertsPerTok: Int?
    public let tieWordEmbeddings: Bool
    /// Width of each dense MLP layer (`dense_intermediate_size`). Falls back to `intermediate_size`.
    public let denseIntermediateSize: Int?
    /// Width of one shared expert (`shared_intermediate_size`). Falls back to `intermediate_size`.
    public let sharedIntermediateSize: Int?
    /// Shared experts per MoE layer (`n_shared_experts`). Their widths are summed.
    public let numSharedExperts: Int
    /// Multiplier on the routed expert sum before the shared expert is added.
    public let routedScalingFactor: Float
    /// SwiGLU-OAI sigmoid slope (`swiglu_alpha`).
    public let swigluAlpha: Float
    /// SwiGLU-OAI clamp (`swiglu_limit`): gate is clamped from above, up on both sides.
    public let swigluLimit: Float
    /// Per-head RMSNorm on queries and keys (`use_qk_norm`).
    public let useQkNorm: Bool
    /// Per-layer 0/1 flags for MoE layers (`moe_layer_freq`). Nil means layers at or after index 3.
    public let moeLayerFreq: [Int]?
    public let sparseAttention: SharpAIMinimaxM3TextSparseAttentionConfiguration?

    /// Attention head width. Falls back to hidden size divided by head count.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / max(attentionHeads, 1))
    }

    /// Number of head channels that receive RoPE (`rotary_dim`, else the full head width).
    public var rotaryDimensions: Int {
        rotaryDim ?? resolvedHeadDim
    }

    /// Width of a dense MLP layer, or nil when neither dense nor base intermediate size is set.
    public var denseWidth: Int? {
        denseIntermediateSize ?? intermediateSize
    }

    /// Whether layer `index` uses the sparse MoE block instead of the dense MLP.
    public func isMoELayer(_ index: Int) -> Bool {
        if let moeLayerFreq {
            return index < moeLayerFreq.count && moeLayerFreq[index] != 0
        }
        return index >= Self.defaultDenseLeadingLayers
    }

    /// Whether layer `index` builds a block indexer and uses block-sparse attention.
    public func isSparseAttentionLayer(_ index: Int) -> Bool {
        guard let sparseAttention, sparseAttention.useSparseAttention else { return false }
        if let freq = sparseAttention.attentionFreq {
            return index < freq.count && freq[index] != 0
        }
        return index >= Self.defaultDenseLeadingLayers
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case rotaryDim = "rotary_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case tieWordEmbeddings = "tie_word_embeddings"
        case denseIntermediateSize = "dense_intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case numSharedExperts = "n_shared_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case swigluAlpha = "swiglu_alpha"
        case swigluLimit = "swiglu_limit"
        case useQkNorm = "use_qk_norm"
        case moeLayerFreq = "moe_layer_freq"
        case sparseAttention = "sparse_attention_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        rotaryDim = try container.decodeIfPresent(Int.self, forKey: .rotaryDim)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        numLocalExperts = try container.decodeIfPresent(Int.self, forKey: .numLocalExperts)
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        denseIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .denseIntermediateSize)
        sharedIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize)
        numSharedExperts =
            try container.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        swigluAlpha = try container.decodeIfPresent(Float.self, forKey: .swigluAlpha) ?? 1.702
        swigluLimit = try container.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 7.0
        useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        moeLayerFreq = try container.decodeIfPresent([Int].self, forKey: .moeLayerFreq)
        sparseAttention = try container.decodeIfPresent(
            SharpAIMinimaxM3TextSparseAttentionConfiguration.self, forKey: .sparseAttention)
    }

    /// Checks the model type, the architecture name when present, and basic shape sanity.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0, kvHeads > 0 else {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "hidden_size, num_hidden_layers, num_attention_heads and num_key_value_heads must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
    }

    /// Checks what the model needs to be built: layer widths, MoE sizes, rotary width and the
    /// sparse indexer shape. `validateModelConfiguration()` runs first.
    public func validateForConstruction() throws {
        try validateModelConfiguration()
        let rotary = rotaryDimensions
        guard rotary > 0, rotary <= resolvedHeadDim, rotary % 2 == 0 else {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "rotary_dim must be an even value in 1...head_dim")
        }
        guard resolvedHeadDim > 0 else {
            throw SharpAIMinimaxM3TextError.invalidConfiguration("head_dim must be positive")
        }
        let hasDenseLayer = (0 ..< hiddenLayers).contains { !isMoELayer($0) }
        if hasDenseLayer, denseWidth == nil {
            throw SharpAIMinimaxM3TextError.invalidConfiguration(
                "dense layers need dense_intermediate_size or intermediate_size")
        }
        let hasMoELayer = (0 ..< hiddenLayers).contains { isMoELayer($0) }
        if hasMoELayer {
            guard let experts = numLocalExperts, let perToken = numExpertsPerTok,
                experts > 0, perToken > 0, perToken <= experts, intermediateSize != nil
            else {
                throw SharpAIMinimaxM3TextError.invalidConfiguration(
                    "MoE layers need num_local_experts, num_experts_per_tok (<= experts) and intermediate_size")
            }
        }
        if let sparse = sparseAttention, sparse.useSparseAttention {
            guard sparse.scoreType == "max" else {
                throw SharpAIMinimaxM3TextError.invalidConfiguration(
                    "sparse_score_type must be max, got \(sparse.scoreType)")
            }
            guard sparse.indexDim > 0, sparse.indexDim % 2 == 0, sparse.indexHeads > 0,
                attentionHeads % sparse.indexHeads == 0, sparse.topkBlocks > 0,
                sparse.blockSize > 0, sparse.localBlocks > 0
            else {
                throw SharpAIMinimaxM3TextError.invalidConfiguration(
                    "sparse_attention_config needs an even index_dim, index heads dividing attention heads, and positive topk, block and local sizes")
            }
        }
    }
}
