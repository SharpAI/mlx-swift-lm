//
//  SharpAISarvamMlaConfiguration.swift
//  MLXLLM
//
//  Text config for `sarvam_mla` (Sarvam 105B, arch `SarvamMLAForCausalLM`).
//  Field names follow the checkpoint config.json in
//  scripts/conformance/offline/fixtures/sarvam_mla.
//  UNVERIFIED: no weights were loaded and modeling_sarvam_moe.py is not in the
//  repo. Defaults for absent keys are assumptions, not checkpoint facts.
//

import Foundation
import MLXLMCommon

enum SharpAISarvamMlaError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "sarvam_mla unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "sarvam_mla invalid configuration: \(message)"
        }
    }
}

struct SharpAISarvamMlaConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?

    // MLA. The fixture has no q_lora_rank, so the query is a full-rank projection.
    var kvLoraRank: Int
    var qkNopeHeadDim: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qHeadDim: Int
    var headDim: Int
    var useQkNorm: Bool

    // MoE.
    var firstKDenseReplace: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var numSharedExperts: Int
    var routedScalingFactor: Float
    var moeRouterEnableExpertBias: Bool

    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qHeadDim = "q_head_dim"
        case headDim = "head_dim"
        case useQkNorm = "use_qk_norm"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)

        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qHeadDim = try c.decode(Int.self, forKey: .qHeadDim)
        headDim = try c.decode(Int.self, forKey: .headDim)
        useQkNorm = try c.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? false

        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension SharpAISarvamMlaConfiguration: ModelConfigurationValidating {
    func validateModelConfiguration() throws {
        func fail(_ message: String) -> SharpAISarvamMlaError {
            .invalidConfiguration(message)
        }

        guard modelType == "sarvam_mla" else {
            throw fail("model_type is \(modelType), expected sarvam_mla")
        }
        guard vocabularySize > 0, hiddenSize > 0, hiddenLayers > 0, attentionHeads > 0 else {
            throw fail("vocab_size, hidden_size, num_hidden_layers and num_attention_heads must be positive")
        }
        guard qkNopeHeadDim + qkRopeHeadDim == qHeadDim else {
            throw fail("q_head_dim must equal qk_nope_head_dim + qk_rope_head_dim")
        }
        guard kvLoraRank + qkRopeHeadDim == headDim else {
            throw fail("head_dim must equal kv_lora_rank + qk_rope_head_dim")
        }
        guard vHeadDim > 0 else {
            throw fail("v_head_dim must be positive")
        }
        guard firstKDenseReplace >= 0, firstKDenseReplace <= hiddenLayers else {
            throw fail("first_k_dense_replace must be in 0...num_hidden_layers")
        }
        if numExperts > 0 {
            guard numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw fail("num_experts_per_tok must be in 1...num_experts")
            }
            guard moeIntermediateSize > 0 else {
                throw fail("moe_intermediate_size must be positive when experts are present")
            }
        }
    }
}
