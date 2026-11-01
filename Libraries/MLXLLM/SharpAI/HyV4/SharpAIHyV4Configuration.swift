//
//  SharpAIHyV4Configuration.swift
//  MLXLLM
//
//  Text config for `hy_v4` (HYV4ForCausalLM). Field names follow the checkpoint
//  config.json, as in scripts/conformance/offline/fixtures/hy_v4/tencent__Hy4-preview.json.
//  UNVERIFIED: no reference forward pass for this architecture is on disk.
//

import Foundation
import MLXLMCommon

enum SharpAIHyV4Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "hy_v4 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "hy_v4 invalid configuration: \(message)"
        }
    }
}

struct SharpAIHyV4RopeParameters: Codable, Sendable {
    var ropeTheta: Float
    var ropeType: String?

    private enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
        case ropeType = "rope_type"
    }
}

struct SharpAIHyV4Configuration: Codable, Sendable, ModelConfigurationValidating {
    var modelType: String
    var architectures: [String]?
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeParameters: SharpAIHyV4RopeParameters?
    var maxPositionEmbeddings: Int?
    var attentionBias: Bool?
    var hiddenAct: String?
    var tieWordEmbeddings: Bool?

    // MLA dimensions.
    var qLoraRank: Int?
    var kvLoraRank: Int?
    var qkNopeHeadDim: Int?
    var qkRopeHeadDim: Int?
    var vHeadDim: Int?
    var gatedMla: Bool?
    var learnableSink: Bool?

    // Shared indexer (sparse attention).
    var indexHeadDim: Int?
    var indexNHeads: Int?
    var indexTopk: Int?
    var indexerTypes: [String]?
    var layerTypes: [String]?

    // MoE.
    var numRoutedExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var numSharedExperts: Int?
    var nGroup: Int?
    var topkGroup: Int?
    var routedScalingFactor: Float?
    var normTopkProb: Bool?
    var gatingType: String?
    var mlpLayerTypes: [String]?
    var swigluLimit: Float?

    // Hyper-connections (iHC).
    var enableIhc: Bool?
    var hcMult: Int?
    var hcMagnitude: Float?
    var hcEps: Float?

    // Multi-token prediction.
    var numNextnPredictLayers: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case gatedMla = "gated_mla"
        case learnableSink = "learnable_sink"
        case indexHeadDim = "index_head_dim"
        case indexNHeads = "index_n_heads"
        case indexTopk = "index_topk"
        case indexerTypes = "indexer_types"
        case layerTypes = "layer_types"
        case numRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "n_shared_experts"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case gatingType = "gating_type"
        case mlpLayerTypes = "mlp_layer_types"
        case swigluLimit = "swiglu_limit"
        case enableIhc = "enable_ihc"
        case hcMult = "hc_mult"
        case hcMagnitude = "hc_magnitude"
        case hcEps = "hc_eps"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    func validateModelConfiguration() throws {
        guard hiddenLayers > 0 else {
            throw SharpAIHyV4Error.invalidConfiguration("num_hidden_layers must be > 0")
        }
        guard ropeParameters != nil else {
            throw SharpAIHyV4Error.invalidConfiguration("rope_parameters is required")
        }
        guard numExpertsPerTok > 0, numExpertsPerTok <= numRoutedExperts else {
            throw SharpAIHyV4Error.invalidConfiguration(
                "num_experts_per_tok must be in 1...n_routed_experts")
        }
        if let indexerTypes = indexerTypes, indexerTypes.count != hiddenLayers {
            throw SharpAIHyV4Error.invalidConfiguration(
                "indexer_types must have one entry per hidden layer")
        }
        if let indexerTypes = indexerTypes, indexerTypes.first == "shared" {
            throw SharpAIHyV4Error.invalidConfiguration(
                "indexer_types[0] must be \"full\": a shared layer needs an earlier full layer")
        }
        if let mlpLayerTypes = mlpLayerTypes, mlpLayerTypes.count != hiddenLayers {
            throw SharpAIHyV4Error.invalidConfiguration(
                "mlp_layer_types must have one entry per hidden layer")
        }
        guard attentionHeads > 0 else {
            throw SharpAIHyV4Error.invalidConfiguration("num_attention_heads must be > 0")
        }
        if let hcMult = hcMult, hcMult <= 0 {
            throw SharpAIHyV4Error.invalidConfiguration("hc_mult must be > 0")
        }
    }
}
