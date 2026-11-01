// SharpAI fork overlay: qwen4_exp configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the config.json fields of Qwen/Qwen3.8-Flash-Next (fixture:
// scripts/conformance/offline/fixtures/qwen4_exp). The meaning of the hyper-connection, PLE,
// n-gram and indexer fields is taken from modeling_qwen4_exp.py in transformers
// (https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
// The model is built from these fields in SharpAIQwen4ExpModel.swift and SharpAIQwen4ExpPLE.swift.

import Foundation
import MLXLMCommon

public enum SharpAIQwen4ExpError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its model class is not implemented yet "
                + "(hyper-connections, PLE, n-gram embeddings and indexer are not written)."
        }
    }
}

public struct SharpAIQwen4ExpRopeParameters: Decodable, Sendable {
    public let ropeType: String
    public let ropeTheta: Double
    public let partialRotaryFactor: Double
    public let mropeInterleaved: Bool
    public let mropeSection: [Int]

    enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case mropeInterleaved = "mrope_interleaved"
        case mropeSection = "mrope_section"
    }
}

public struct SharpAIQwen4ExpMTPConfiguration: Decodable, Sendable {
    public let hybrid: Bool?
    public let layerTypes: [String]?
    public let numHiddenLayers: Int?
    public let ropeTheta: Double?
    public let mtpUseHiddenStateFromLayer: Int?

    enum CodingKeys: String, CodingKey {
        case hybrid
        case layerTypes = "layer_types"
        case numHiddenLayers = "num_hidden_layers"
        case ropeTheta = "rope_theta"
        case mtpUseHiddenStateFromLayer = "mtp_use_hidden_state_from_layer"
    }
}

public struct SharpAIQwen4ExpTextConfiguration: Decodable, Sendable {
    public let modelType: String?
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Double
    public let hiddenAct: String?
    public let attentionBias: Bool?
    public let maxPositionEmbeddings: Int?
    public let partialRotaryFactor: Double?
    public let layerTypes: [String]
    public let fullAttentionInterval: Int
    public let linearConvKernelDim: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let mambaSsmDtype: String?
    /// "silu" or "sigmoid". Selects the GDN output gate: the reference's RMSNormGated uses
    /// ACT2FN[output_gate_type] (Verified against modeling_qwen4_exp.py; both cases are ported).
    public let outputGateType: String?
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let ropeParameters: SharpAIQwen4ExpRopeParameters

    // Hyper-connections (built in SharpAIQwen4ExpModel.swift and SharpAIQwen4ExpPLE.swift).
    public let hcCount: Int?
    public let hcLowrank: Int?

    // N-gram embedding (built in SharpAIQwen4ExpModel.swift and SharpAIQwen4ExpPLE.swift).
    public let ngramSize: Int?
    public let ngramVocabSizeBase: Int?
    public let headsPerNgram: Int?
    public let splitNgramParts: Int?
    public let makeNgramVocabSizeDivisibleBy: Int?

    // Per-layer embedding (PLE) (built in SharpAIQwen4ExpModel.swift and SharpAIQwen4ExpPLE.swift).
    public let pleEmbedDim: Int?
    public let pleConvKernelSize: Int?
    public let pleLayerIds: [Int]?

    // Sparse indexer (built in SharpAIQwen4ExpModel.swift and SharpAIQwen4ExpPLE.swift).
    public let indexerBudget: Int?
    public let indexerCompressRatio: Int?
    public let indexerHeadDim: Int?
    public let indexerKvHeads: Int?
    public let indexerNHeads: Int?

    // MTP head (MTP head not ported: no reference class in transformers).
    public let mtpNumHiddenLayers: Int?
    public let mtpUseDedicatedEmbeddings: Bool?
    public let mtp: SharpAIQwen4ExpMTPConfiguration?

    public let bosTokenId: Int?
    public let eosTokenId: Int?
    public let padTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case mambaSsmDtype = "mamba_ssm_dtype"
        case outputGateType = "output_gate_type"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case ropeParameters = "rope_parameters"
        case hcCount = "hc_count"
        case hcLowrank = "hc_lowrank"
        case ngramSize = "ngram_size"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case headsPerNgram = "heads_per_ngram"
        case splitNgramParts = "split_ngram_parts"
        case makeNgramVocabSizeDivisibleBy = "make_ngram_vocab_size_divisible_by"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case pleLayerIds = "ple_layer_ids"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case indexerHeadDim = "indexer_head_dim"
        case indexerKvHeads = "indexer_kv_heads"
        case indexerNHeads = "indexer_n_heads"
        case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
        case mtp
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }
}

public struct SharpAIQwen4ExpVisionConfiguration: Decodable, Sendable {
    public let modelType: String?
    public let depth: Int?
    public let hiddenSize: Int?
    public let intermediateSize: Int?
    public let numHeads: Int?
    public let inChannels: Int?
    public let patchSize: Int?
    public let temporalPatchSize: Int?
    public let spatialMergeSize: Int?
    public let outHiddenSize: Int?
    public let numPositionEmbeddings: Int?
    public let hiddenAct: String?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case depth
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHeads = "num_heads"
        case inChannels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case outHiddenSize = "out_hidden_size"
        case numPositionEmbeddings = "num_position_embeddings"
        case hiddenAct = "hidden_act"
    }
}

public struct SharpAIQwen4ExpConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "qwen4_exp"
    public static let expectedLayerTypes = ["linear_attention", "full_attention"]

    public let modelType: String
    public let architectures: [String]?
    public let textConfig: SharpAIQwen4ExpTextConfiguration
    public let visionConfig: SharpAIQwen4ExpVisionConfiguration?
    public let imageTokenId: Int?
    public let videoTokenId: Int?
    public let visionStartTokenId: Int?
    public let visionEndTokenId: Int?
    public let tieWordEmbeddings: Bool?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        let layerTypes = textConfig.layerTypes
        guard layerTypes.count == textConfig.numHiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: layer_types has \(layerTypes.count) entries, "
                + "num_hidden_layers is \(textConfig.numHiddenLayers)")
        }
        for layerType in layerTypes where !Self.expectedLayerTypes.contains(layerType) {
            throw ModelFactoryError.invalidConfiguration(
                "qwen4_exp: unknown layer type \(layerType)")
        }
    }
}
