//
//  SharpAIParam2moeConfiguration.swift
//  MLXLLM
//
//  Text config for `param2moe` (bharatgenai/Param2-17B-A2.4B-Thinking). Field names follow the
//  checkpoint config.json in scripts/conformance/offline/fixtures/param2moe, as read by vLLM
//  param2moe.py.
//  UNVERIFIED: no checkpoint weights were loaded. Defaults for absent keys follow vLLM getattr defaults.
//

import Foundation
import MLXLMCommon

enum SharpAIParam2moeError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "param2moe unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "param2moe invalid configuration: \(message)"
        }
    }
}

struct SharpAIParam2moeConfiguration: Decodable, Sendable {
    var modelType: String
    var architectures: [String]
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var partialRotaryFactor: Float
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var useBias: Bool
    var useQKVBias: Bool
    var useQKNorm: Bool

    // MoE layout.
    var firstKDenseReplace: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var numSharedExperts: Int
    var moeSharedExpertIntermediateSize: Int
    var moeRouterEnableExpertBias: Bool
    var routedScalingFactor: Float
    var scoreFunction: String
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBias = "use_bias"
        case useQKVBias = "use_qkv_bias"
        case useQKNorm = "use_qk_norm"
        case firstKDenseReplace = "first_k_dense_replace"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
    }

    /// `rope_theta` and `partial_rotary_factor` may sit in a nested `rope_parameters` dict.
    private struct RopeParameters: Decodable {
        var ropeTheta: Float?
        var partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "param2moe"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / max(attentionHeads, 1))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768

        let topLevelTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = topLevelTheta ?? nested?.ropeTheta ?? 10000

        let topLevelPartial = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        partialRotaryFactor = topLevelPartial ?? nested?.partialRotaryFactor ?? 1.0

        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        useQKVBias = try c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false

        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? intermediateSize
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1
        // vLLM reads this key as the total shared width. It is not multiplied again.
        moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)
            ?? (moeIntermediateSize * numSharedExperts)
        moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        scoreFunction =
            try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
    }

    /// Number of head channels that receive RoPE.
    var rotaryDimensions: Int {
        Int(Float(headDim) * partialRotaryFactor)
    }

    /// The first `first_k_dense_replace` layers use a dense MLP. Later layers are sparse.
    func isSparseLayer(_ layer: Int) -> Bool {
        numExperts > 0 && layer >= firstKDenseReplace
    }

    func validateModelConfiguration() throws {
        if !architectures.isEmpty, !architectures.contains("Param2MoEForCausalLM") {
            throw SharpAIParam2moeError.unsupported("architectures \(architectures)")
        }
        guard vocabularySize > 0, hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0 else {
            throw SharpAIParam2moeError.invalidConfiguration("sizes must be positive")
        }
        guard attentionHeads > 0, kvHeads > 0, headDim > 0,
            attentionHeads % kvHeads == 0
        else {
            throw SharpAIParam2moeError.invalidConfiguration(
                "attention heads must be a multiple of kv heads")
        }
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1, rotaryDimensions > 0 else {
            throw SharpAIParam2moeError.invalidConfiguration("partial_rotary_factor out of range")
        }
        guard ["sigmoid", "softmax"].contains(scoreFunction) else {
            throw SharpAIParam2moeError.unsupported("score_function \(scoreFunction)")
        }
        if numExperts > 0 {
            // UNVERIFIED: the expert-bias-off path is not implemented. vLLM handles it with no bias.
            guard moeRouterEnableExpertBias else {
                throw SharpAIParam2moeError.unsupported("moe_router_enable_expert_bias=false")
            }
            guard numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw SharpAIParam2moeError.invalidConfiguration(
                    "num_experts_per_tok must be in 1...num_experts")
            }
            guard nGroup > 0, numExperts % nGroup == 0 else {
                throw SharpAIParam2moeError.invalidConfiguration(
                    "n_group must divide num_experts")
            }
            guard topkGroup > 0, topkGroup <= nGroup else {
                throw SharpAIParam2moeError.invalidConfiguration("topk_group must be in 1...n_group")
            }
            if nGroup > 1 {
                let groupSize = numExperts / nGroup
                guard groupSize >= 2, numExpertsPerTok <= topkGroup * groupSize else {
                    throw SharpAIParam2moeError.invalidConfiguration(
                        "each expert group needs at least 2 experts, and num_experts_per_tok must be <= topk_group * (num_experts / n_group)"
                    )
                }
            }
        }
    }
}
