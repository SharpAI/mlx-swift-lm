// SharpAI fork overlay: deepseek_v3_bidirec configuration. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: no reference source for deepseek_v3_bidirec was found in the repo, the vLLM
// copy, or on this machine. Field names follow DeepseekV3Configuration. Defaults follow the
// DeepSeek-V3 HF config. The one deliberate change: q_lora_rank is optional here, and null
// or absent selects the direct q_proj path.

import Foundation
import MLXLMCommon

struct SharpAIDeepseekV3BidirecConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    let modelType: String
    let vocabSize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let moeIntermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let nSharedExperts: Int?
    let nRoutedExperts: Int?
    let routedScalingFactor: Float
    let kvLoraRank: Int
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let normTopkProb: Bool
    let nGroup: Int?
    let topkGroup: Int?
    let numExpertsPerTok: Int?
    let moeLayerFreq: Int
    let firstKDenseReplace: Int
    let maxPositionEmbeddings: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let ropeScaling: [String: StringOrNumber]?
    let attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v3_bidirec"
        let vocab = try c.decode(Int.self, forKey: .vocabSize)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let intermediate = try c.decode(Int.self, forKey: .intermediateSize)
        let heads = try c.decode(Int.self, forKey: .numAttentionHeads)
        vocabSize = vocab
        hiddenSize = hidden
        intermediateSize = intermediate
        numAttentionHeads = heads
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? heads
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? intermediate

        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup)
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
    }

    func validateModelConfiguration() throws {
        let positive = [
            vocabSize, hiddenSize, intermediateSize, moeIntermediateSize, numHiddenLayers,
            numAttentionHeads, numKeyValueHeads, kvLoraRank, qkRopeHeadDim, qkNopeHeadDim,
            vHeadDim,
        ]
        guard positive.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v3_bidirec: sizes, heads and ranks must be positive")
        }
        if let qLoraRank, qLoraRank <= 0 {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v3_bidirec: q_lora_rank must be positive when set")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v3_bidirec: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard qkRopeHeadDim % 2 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v3_bidirec: qk_rope_head_dim must be even")
        }
        guard firstKDenseReplace >= 0, moeLayerFreq > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_v3_bidirec: first_k_dense_replace must be >= 0 and moe_layer_freq > 0")
        }
        if let nRoutedExperts {
            let groups = nGroup ?? 1
            let topk = topkGroup ?? 1
            guard nRoutedExperts > 0, groups > 0, nRoutedExperts % groups == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "deepseek_v3_bidirec: n_routed_experts must be a positive multiple of n_group")
            }
            guard groups == 1 || nRoutedExperts / groups >= 2 else {
                throw ModelFactoryError.invalidConfiguration(
                    "deepseek_v3_bidirec: each expert group needs at least 2 experts")
            }
            guard topk >= 1, topk <= groups else {
                throw ModelFactoryError.invalidConfiguration(
                    "deepseek_v3_bidirec: topk_group must be in 1...n_group")
            }
            let perTok = numExpertsPerTok ?? 1
            guard perTok >= 1, perTok <= nRoutedExperts else {
                throw ModelFactoryError.invalidConfiguration(
                    "deepseek_v3_bidirec: num_experts_per_tok must be in 1...n_routed_experts")
            }
        }
    }
}
