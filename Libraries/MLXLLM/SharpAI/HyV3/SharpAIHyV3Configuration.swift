//
//  SharpAIHyV3Configuration.swift
//  MLXLLM
//
//  Text config for `hy_v3` (Tencent Hy3). Field names follow the checkpoint
//  config.json in scripts/conformance/offline/fixtures/hy_v3 (tencent/Hy3 config.json,
//  every CodingKeys entry matches it). UNVERIFIED: defaults for keys absent from the
//  checkpoint config are assumptions, and no checkpoint weights were loaded.
//

import Foundation
import MLXLMCommon

enum SharpAIHyV3Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "hy_v3 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "hy_v3 invalid configuration: \(message)"
        }
    }
}

struct SharpAIHyV3Configuration: Decodable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeType: String
    var maxPositionEmbeddings: Int?
    var qkNorm: Bool
    var attentionBias: Bool

    // MoE.
    var firstKDenseReplace: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var numSharedExperts: Int
    var moeRouterUseSigmoid: Bool
    var moeRouterEnableExpertBias: Bool
    var routeNorm: Bool
    var routerScalingFactor: Float

    // Layout.
    var tieWordEmbeddings: Bool
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case qkNorm = "qk_norm"
        case attentionBias = "attention_bias"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case moeRouterUseSigmoid = "moe_router_use_sigmoid"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case routeNorm = "route_norm"
        case routerScalingFactor = "router_scaling_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    /// The nested `rope_parameters` object of the checkpoint config.
    private struct RopeParameters: Decodable {
        var ropeTheta: Float
        var ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hy_v3"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = hidden
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = heads
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        // Fallback when head_dim is absent: hidden size split across heads.
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / max(heads, 1))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5

        if let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters) {
            ropeTheta = rope.ropeTheta
            ropeType = rope.ropeType ?? "default"
        } else {
            ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
            ropeType = "default"
        }
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        qkNorm = try c.decodeIfPresent(Bool.self, forKey: .qkNorm) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        moeRouterUseSigmoid =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterUseSigmoid) ?? true
        moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? false
        routeNorm = try c.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true
        routerScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routerScalingFactor) ?? 1.0

        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
    }
}

extension SharpAIHyV3Configuration: ModelConfigurationValidating {

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0 else {
            throw SharpAIHyV3Error.invalidConfiguration(
                "hidden_size, num_hidden_layers and vocab_size must be > 0")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIHyV3Error.invalidConfiguration(
                "num_key_value_heads must divide num_attention_heads")
        }
        guard headDim > 0, headDim % 2 == 0 else {
            throw SharpAIHyV3Error.invalidConfiguration("head_dim must be a positive even number")
        }
        guard (0 ... hiddenLayers).contains(firstKDenseReplace) else {
            throw SharpAIHyV3Error.invalidConfiguration(
                "first_k_dense_replace must be in 0...num_hidden_layers")
        }
        guard numExperts > 0, numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
            throw SharpAIHyV3Error.invalidConfiguration(
                "num_experts_per_tok must be in 1...num_experts")
        }
        guard numSharedExperts >= 0 else {
            throw SharpAIHyV3Error.invalidConfiguration("num_shared_experts must be >= 0")
        }
        guard ropeType == "default" else {
            throw SharpAIHyV3Error.unsupported("rope_type=\(ropeType)")
        }
        guard moeRouterUseSigmoid else {
            throw SharpAIHyV3Error.unsupported("softmax MoE router (moe_router_use_sigmoid=false)")
        }
        guard moeRouterEnableExpertBias else {
            throw SharpAIHyV3Error.unsupported(
                "router without expert bias (moe_router_enable_expert_bias=false)")
        }
    }
}
