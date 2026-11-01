//
//  SharpAISarvamMoeConfiguration.swift
//  MLXLLM
//
//  Text config for `sarvam_moe` (Sarvam-30B, SarvamMoEForCausalLM).
//  Field names follow the checkpoint config.json. The field set matches
//  scripts/conformance/offline/fixtures/sarvam_moe/sarvamai__sarvam-30b.json.
//  Field names and defaults were checked against sarvamai/sarvam-30b's config.json and
//  modeling_sarvam_moe.py (the `score_function`/`n_group`/`topk_group` router semantics).
//

import Foundation
import MLXLMCommon

enum SharpAISarvamMoeError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "sarvam_moe unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "sarvam_moe invalid configuration: \(message)"
        }
    }
}

struct SharpAISarvamMoeConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int?
    var useQKVBias: Bool
    var useBias: Bool
    var useQKNorm: Bool
    var tieWordEmbeddings: Bool

    // MoE.
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var numSharedExperts: Int
    var moeSharedExpertIntermediateSize: Int
    var firstKDenseReplace: Int
    var nGroup: Int
    var topkGroup: Int
    var routedScalingFactor: Float
    var normTopkProb: Bool
    var scoreFunction: String
    var moeRouterEnableExpertBias: Bool

    /// Rotary dimension within each head.
    var ropeDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case useQKVBias = "use_qkv_bias"
        case useBias = "use_bias"
        case useQKNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case scoreFunction = "score_function"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "sarvam_moe"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = hidden
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / heads)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        useQKVBias = try c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)
            ?? moeIntermediateSize
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        scoreFunction = try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? false
    }
}

extension SharpAISarvamMoeConfiguration: ModelConfigurationValidating {

    func validateModelConfiguration() throws {
        guard scoreFunction == "sigmoid" else {
            throw SharpAISarvamMoeError.unsupported("score_function \(scoreFunction)")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAISarvamMoeError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard headDim > 0, ropeDim > 0, ropeDim <= headDim, ropeDim % 2 == 0 else {
            throw SharpAISarvamMoeError.invalidConfiguration(
                "rotary dim \(ropeDim) must be even and within head_dim \(headDim)")
        }
        guard numExperts > 0, numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
            throw SharpAISarvamMoeError.invalidConfiguration(
                "num_experts_per_tok must be in 1...num_experts")
        }
        guard nGroup > 0, numExperts % nGroup == 0, numExperts / nGroup >= 2 else {
            throw SharpAISarvamMoeError.invalidConfiguration(
                "n_group must divide num_experts into groups of at least 2")
        }
        guard topkGroup > 0, topkGroup <= nGroup else {
            throw SharpAISarvamMoeError.invalidConfiguration("topk_group must be in 1...n_group")
        }
    }
}
