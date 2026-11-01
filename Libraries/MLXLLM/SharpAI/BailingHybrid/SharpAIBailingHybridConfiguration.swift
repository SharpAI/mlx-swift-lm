// SharpAI fork overlay: bailing_hybrid configuration. Not part of upstream mlx-swift-lm.
//
// Reference: mlx_lm/models/bailing_hybrid.py (ml-explore/mlx-lm, PR #1227, class BailingMoeV2_5ForCausalLM),
// and the config.json of inclusionAI/Ling-2.5-1T, which uses model_type "bailing_hybrid".
// Key names and defaults follow ModelArgs in that file. Fields the model needs but the Python
// ModelArgs requires (MLA dims, layer_group_size, group_norm_size) are optional here so that the
// minimal configs in the existing tests still decode; validateForForward() enforces them.
// Not covered: the Ling 3.x KDA variant (BailingMoeV3ForCausalLM), rope_scaling (decoded as null only),
// and per-layer swiglu clamp.

import Foundation
import MLXLMCommon

public enum SharpAIBailingHybridError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) is registered, but its forward pass is not implemented."
        }
    }
}

public struct SharpAIBailingHybridConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    // MoE fields.
    public let numExperts: Int?
    public let numExpertsPerTok: Int?
    public let moeIntermediateSize: Int?
    public let numSharedExperts: Int?
    public let firstKDenseReplace: Int?
    public let numNextnPredictLayers: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let routedScalingFactor: Float
    public let scoreFunction: String
    public let normTopkProb: Bool
    public let moeRouterEnableExpertBias: Bool
    public let moeSharedExpertIntermediateSize: Int?

    // Attention fields.
    public let headDim: Int?
    public let qLoraRank: Int?
    public let kvLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let vHeadDim: Int?
    public let layerGroupSize: Int?
    public let groupNormSize: Int?
    public let partialRotaryFactor: Float
    public let ropeInterleave: Bool
    public let useQKNorm: Bool
    public let useQKVBias: Bool
    public let useBias: Bool
    public let maxPositionEmbeddings: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case firstKDenseReplace = "first_k_dense_replace"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case normTopkProb = "norm_topk_prob"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case headDim = "head_dim"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case layerGroupSize = "layer_group_size"
        case groupNormSize = "group_norm_size"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeInterleave = "rope_interleave"
        case useQKNorm = "use_qk_norm"
        case useQKVBias = "use_qkv_bias"
        case useBias = "use_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "bailing_hybrid"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts)
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace)
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
        // Python ModelArgs requires n_group and topk_group. Defaults here disable group pruning.
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? nGroup
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        scoreFunction = try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid"
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        moeRouterEnableExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? true
        moeSharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeSharedExpertIntermediateSize)

        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
        layerGroupSize = try c.decodeIfPresent(Int.self, forKey: .layerGroupSize)
        groupNormSize = try c.decodeIfPresent(Int.self, forKey: .groupNormSize)
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
        useQKVBias = try c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
    }

    /// Checks the fields that decoding does not require. Shared by all callers.
    public func validateModelConfiguration() throws {
        let dims = [hiddenSize, hiddenLayers, attentionHeads, kvHeads, intermediateSize, vocabularySize]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: hidden, layer, head, intermediate and vocab sizes must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard numNextnPredictLayers >= 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: num_nextn_predict_layers must not be negative")
        }
        guard scoreFunction == "sigmoid" || scoreFunction == "softmax" else {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: score_function must be sigmoid or softmax")
        }
        let resolvedHeadDim = headDim ?? hiddenSize / attentionHeads
        let rotary = Int(Float(resolvedHeadDim) * partialRotaryFactor)
        guard rotary > 0, rotary <= resolvedHeadDim, rotary % 2 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: invalid rotary dimension \(rotary) for head_dim \(resolvedHeadDim)")
        }
    }

    /// Checks what the model builder needs: MLA dims, layer grouping, and MoE routing.
    /// Call before building the model. Throws instead of trapping on a bad config.
    public func validateForForward() throws {
        try validateModelConfiguration()
        let required: [(String, Int?)] = [
            ("kv_lora_rank", kvLoraRank),
            ("qk_nope_head_dim", qkNopeHeadDim),
            ("qk_rope_head_dim", qkRopeHeadDim),
            ("v_head_dim", vHeadDim),
            ("layer_group_size", layerGroupSize),
            ("group_norm_size", groupNormSize),
        ]
        for (name, value) in required {
            guard let value, value > 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "bailing_hybrid: \(name) is required and must be positive")
            }
        }
        if let groupNormSize, (attentionHeads * (headDim ?? hiddenSize / attentionHeads)) % groupNormSize != 0 {
            throw ModelFactoryError.invalidConfiguration(
                "bailing_hybrid: group_norm_size must divide attention_heads * head_dim")
        }
        if let numExperts {
            guard numExperts > 0, let perTok = numExpertsPerTok, perTok > 0, perTok <= numExperts,
                let moeInter = moeIntermediateSize, moeInter > 0
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "bailing_hybrid: MoE needs num_experts_per_tok <= num_experts and moe_intermediate_size > 0")
            }
            guard nGroup > 0, numExperts % nGroup == 0, topkGroup >= 0, topkGroup <= nGroup else {
                throw ModelFactoryError.invalidConfiguration(
                    "bailing_hybrid: n_group must divide num_experts and topk_group must be <= n_group")
            }
        }
    }

    /// Mirrors DecoderLayer.is_global in bailing_hybrid.py. Requires a validated layer_group_size.
    public func isGlobalLayer(_ layerIdx: Int) -> Bool {
        let group = layerGroupSize ?? 1
        return (layerIdx + 1) % group == 0 || layerIdx >= (hiddenLayers / group) * group
    }

    /// Mirrors the MLP/MoE choice in DecoderLayer.__init__.
    public func isMoELayer(_ layerIdx: Int) -> Bool {
        numExperts != nil && layerIdx >= (firstKDenseReplace ?? 0)
    }
}
