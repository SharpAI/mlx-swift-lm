// SharpAI fork overlay: deepseek_v41 configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow the real checkpoint config.json of
// deepseek-ai/DeepSeek-V4.1-Flash-1017 (text_config) and the ModelArgs defaults in
// ml-explore/mlx-lm mlx_lm/models/deepseek_v41.py.
// Verified against: https://raw.githubusercontent.com/ml-explore/mlx-lm/main/mlx_lm/models/deepseek_v41.py
// Verified against: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/raw/main/config.json
// Vision tower fields and the CED config are not mapped yet (the vision tower is not ported).

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV41Error: LocalizedError {
    case forwardNotImplemented(String)
    case missingTextConfig
    case engramNotSupported([Int])

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) is registered, but its model class is not implemented."
        case .missingTextConfig:
            return "deepseek_v41 config has no text_config section, so no text model can be built."
        case .engramNotSupported(let layers):
            return
                "Layers \(layers) use Engram memory, which this implementation does not have: "
                + "the n-gram hash tables, the hashed embedding lookup and the engram attention are missing."
        }
    }
}

/// The `rope_scaling` section of text_config (YaRN for deepseek_v41). Missing keys take the
/// mlx-lm defaults: factor 1.0, beta_fast 32, beta_slow 1.
public struct SharpAIDeepseekV41RopeScaling: Decodable, Sendable {
    public let factor: Float
    public let originalMaxPositionEmbeddings: Int?
    public let betaFast: Float
    public let betaSlow: Float

    enum CodingKeys: String, CodingKey {
        case factor
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        factor = try c.decodeIfPresent(Float.self, forKey: .factor) ?? 1.0
        originalMaxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings)
        betaFast = try c.decodeIfPresent(Float.self, forKey: .betaFast) ?? 32
        betaSlow = try c.decodeIfPresent(Float.self, forKey: .betaSlow) ?? 1
    }
}

/// The `text_config` section of a deepseek_v41 checkpoint. Missing fields take the
/// mlx-lm `ModelArgs` defaults; fields with no default in that reference are optional.
public struct SharpAIDeepseekV41TextConfiguration: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let headDim: Int
    public let qkRopeHeadDim: Int
    public let qLoraRank: Int
    public let oLoraRank: Int
    public let oGroups: Int
    public let moeIntermediateSize: Int
    public let nRoutedExperts: Int
    public let nSharedExperts: Int?
    public let numExpertsPerTok: Int
    public let routedScalingFactor: Double
    public let rmsNormEps: Float
    public let swigluLimit: Float
    public let scoringFunc: String
    public let gateTemp: Float
    public let normTopkProb: Bool
    public let slidingWindow: Int
    /// Per-layer compress ratios exactly as written in the checkpoint (0 = window only).
    public let compressRatios: [Int]
    public let kvSourceLayerIds: [Int]
    public let indexSourceLayerIds: [Int]
    public let indexNHeads: Int
    public let indexHeadDim: Int
    public let indexTopk: Int
    public let candidateSourceLayerId: Int
    public let candidateTopkBlocks: Int
    public let candidateBlockSize: Int
    public let hcMult: Int
    public let hcSinkhornIters: Int
    public let hcEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let ropeScaling: SharpAIDeepseekV41RopeScaling?
    public let compressRopeTheta: Float
    public let engramLayerIds: [Int]
    public let engramMaxNgramSize: Int?
    public let engramVocabSize: Int?
    public let engramNHeads: Int?
    public let engramHeadDim: Int?
    public let numNextnPredictLayers: Int?
    public let dsparkBlockSize: Int?
    public let dsparkTargetLayerIds: [Int]
    public let dsparkMarkovRank: Int?
    public let dsparkNRoutedExperts: Int?
    public let dsparkNumExpertsPerTok: Int?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qLoraRank = "q_lora_rank"
        case oLoraRank = "o_lora_rank"
        case oGroups = "o_groups"
        case moeIntermediateSize = "moe_intermediate_size"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case rmsNormEps = "rms_norm_eps"
        case swigluLimit = "swiglu_limit"
        case scoringFunc = "scoring_func"
        case gateTemp = "gate_temp"
        case normTopkProb = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case compressRatios = "compress_ratios"
        case kvSourceLayerIds = "kv_source_layer_ids"
        case indexSourceLayerIds = "index_source_layer_ids"
        case indexNHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopk = "index_topk"
        case candidateSourceLayerId = "candidate_source_layer_id"
        case candidateTopkBlocks = "candidate_topk_blocks"
        case candidateBlockSize = "candidate_block_size"
        case hcMult = "hc_mult"
        case hcSinkhornIters = "hc_sinkhorn_iters"
        case hcEps = "hc_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case compressRopeTheta = "compress_rope_theta"
        case engramLayerIds = "engram_layer_ids"
        case engramMaxNgramSize = "engram_max_ngram_size"
        case engramVocabSize = "engram_vocab_size"
        case engramNHeads = "engram_n_heads"
        case engramHeadDim = "engram_head_dim"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case dsparkBlockSize = "dspark_block_size"
        case dsparkTargetLayerIds = "dspark_target_layer_ids"
        case dsparkMarkovRank = "dspark_markov_rank"
        case dsparkNRoutedExperts = "dspark_n_routed_experts"
        case dsparkNumExpertsPerTok = "dspark_num_experts_per_tok"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 129_280
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 64
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 512
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank) ?? 1280
        oLoraRank = try c.decodeIfPresent(Int.self, forKey: .oLoraRank) ?? 1024
        oGroups = try c.decodeIfPresent(Int.self, forKey: .oGroups) ?? 8
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 2304
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 384
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        routedScalingFactor =
            try c.decodeIfPresent(Double.self, forKey: .routedScalingFactor) ?? 1.5
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-20
        swigluLimit = try c.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 10.0
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "sqrtsoftplus"
        gateTemp = try c.decodeIfPresent(Float.self, forKey: .gateTemp) ?? 1.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        compressRatios = try c.decodeIfPresent([Int].self, forKey: .compressRatios) ?? []
        kvSourceLayerIds = try c.decodeIfPresent([Int].self, forKey: .kvSourceLayerIds) ?? []
        indexSourceLayerIds =
            try c.decodeIfPresent([Int].self, forKey: .indexSourceLayerIds) ?? []
        indexNHeads = try c.decodeIfPresent(Int.self, forKey: .indexNHeads) ?? 32
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopk = try c.decodeIfPresent(Int.self, forKey: .indexTopk) ?? 512
        candidateSourceLayerId =
            try c.decodeIfPresent(Int.self, forKey: .candidateSourceLayerId) ?? -1
        candidateTopkBlocks = try c.decodeIfPresent(Int.self, forKey: .candidateTopkBlocks) ?? 2048
        candidateBlockSize = try c.decodeIfPresent(Int.self, forKey: .candidateBlockSize) ?? 8
        hcMult = try c.decodeIfPresent(Int.self, forKey: .hcMult) ?? 4
        hcSinkhornIters = try c.decodeIfPresent(Int.self, forKey: .hcSinkhornIters) ?? 20
        hcEps = try c.decodeIfPresent(Float.self, forKey: .hcEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try c.decodeIfPresent(SharpAIDeepseekV41RopeScaling.self, forKey: .ropeScaling)
        compressRopeTheta =
            try c.decodeIfPresent(Float.self, forKey: .compressRopeTheta) ?? 160_000
        engramLayerIds = try c.decodeIfPresent([Int].self, forKey: .engramLayerIds) ?? []
        engramMaxNgramSize = try c.decodeIfPresent(Int.self, forKey: .engramMaxNgramSize)
        engramVocabSize = try c.decodeIfPresent(Int.self, forKey: .engramVocabSize)
        engramNHeads = try c.decodeIfPresent(Int.self, forKey: .engramNHeads)
        engramHeadDim = try c.decodeIfPresent(Int.self, forKey: .engramHeadDim)
        numNextnPredictLayers = try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers)
        dsparkBlockSize = try c.decodeIfPresent(Int.self, forKey: .dsparkBlockSize)
        dsparkTargetLayerIds =
            try c.decodeIfPresent([Int].self, forKey: .dsparkTargetLayerIds) ?? []
        dsparkMarkovRank = try c.decodeIfPresent(Int.self, forKey: .dsparkMarkovRank)
        dsparkNRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .dsparkNRoutedExperts)
        dsparkNumExpertsPerTok =
            try c.decodeIfPresent(Int.self, forKey: .dsparkNumExpertsPerTok)
    }
}

public struct SharpAIDeepseekV41Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "deepseek_v41"

    public let modelType: String
    public let architectures: [String]
    /// The text sub-model type, expected to be "deepseek_v41_text" (checkpoint text_config).
    public let textModelType: String?
    /// The vision sub-model type, expected to be "deepseek_v41_vision" (checkpoint vision_config).
    public let visionModelType: String?
    /// Full text_config fields. Nil when the checkpoint has no text_config section.
    public let text: SharpAIDeepseekV41TextConfiguration?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    private struct SubConfig: Decodable {
        let modelType: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textModelType = try c.decodeIfPresent(SubConfig.self, forKey: .textConfig)?.modelType
        visionModelType = try c.decodeIfPresent(SubConfig.self, forKey: .visionConfig)?.modelType
        text = try c.decodeIfPresent(SharpAIDeepseekV41TextConfiguration.self, forKey: .textConfig)
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v41: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
    }
}
