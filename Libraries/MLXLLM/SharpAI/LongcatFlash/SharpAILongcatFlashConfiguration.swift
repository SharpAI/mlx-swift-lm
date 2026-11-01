// SharpAI fork overlay: longcat_flash configuration (LongCat-Flash-Lite, n-gram variant).
// Field names follow mlx-lm's longcat_flash_ngram.ModelArgs. Not part of upstream.

import Foundation
import MLXLMCommon

enum SharpAILongcatFlashError: LocalizedError {
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason):
            "Invalid longcat_flash configuration: \(reason)"
        }
    }
}

struct SharpAILongcatFlashConfiguration: Codable, Sendable {
    var modelType: String
    var vocabSize: Int
    var hiddenSize: Int
    var ffnHiddenSize: Int
    var expertFfnHiddenSize: Int
    var numLayers: Int
    var numAttentionHeads: Int
    var kvLoraRank: Int
    var qLoraRank: Int
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var mlaScaleQLora: Bool
    var mlaScaleKvLora: Bool
    var routedScalingFactor: Float
    var nRoutedExperts: Int
    var zeroExpertNum: Int
    var zeroExpertType: String
    var moeTopk: Int
    var normTopkProb: Bool
    var routerBias: Bool
    var attentionBias: Bool
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var rmsNormEps: Float
    var ngramVocabSizeRatio: Int
    var embNeighborNum: Int
    var embSplitNum: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case ffnHiddenSize = "ffn_hidden_size"
        case expertFfnHiddenSize = "expert_ffn_hidden_size"
        case numLayers = "num_layers"
        case numAttentionHeads = "num_attention_heads"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case mlaScaleQLora = "mla_scale_q_lora"
        case mlaScaleKvLora = "mla_scale_kv_lora"
        case routedScalingFactor = "routed_scaling_factor"
        case nRoutedExperts = "n_routed_experts"
        case zeroExpertNum = "zero_expert_num"
        case zeroExpertType = "zero_expert_type"
        case moeTopk = "moe_topk"
        case normTopkProb = "norm_topk_prob"
        case routerBias = "router_bias"
        case attentionBias = "attention_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case rmsNormEps = "rms_norm_eps"
        case ngramVocabSizeRatio = "ngram_vocab_size_ratio"
        case embNeighborNum = "emb_neighbor_num"
        case embSplitNum = "emb_split_num"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "longcat_flash"
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        ffnHiddenSize = try c.decode(Int.self, forKey: .ffnHiddenSize)
        expertFfnHiddenSize = try c.decode(Int.self, forKey: .expertFfnHiddenSize)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decode(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        mlaScaleQLora = try c.decode(Bool.self, forKey: .mlaScaleQLora)
        mlaScaleKvLora = try c.decode(Bool.self, forKey: .mlaScaleKvLora)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        // UNVERIFIED: the Lite fixture omits n_routed_experts. 512 is the HF LongcatFlashConfig default; confirm against the real Lite config.
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 512
        zeroExpertNum = try c.decode(Int.self, forKey: .zeroExpertNum)
        zeroExpertType = try c.decodeIfPresent(String.self, forKey: .zeroExpertType) ?? "identity"
        moeTopk = try c.decode(Int.self, forKey: .moeTopk)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        routerBias = try c.decodeIfPresent(Bool.self, forKey: .routerBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ngramVocabSizeRatio =
            try c.decodeIfPresent(Int.self, forKey: .ngramVocabSizeRatio) ?? 78
        embNeighborNum = try c.decodeIfPresent(Int.self, forKey: .embNeighborNum) ?? 4
        embSplitNum = try c.decodeIfPresent(Int.self, forKey: .embSplitNum) ?? 4
    }
}

extension SharpAILongcatFlashConfiguration {
    /// Size of the hashed n-gram table, `m` in mlx-lm.
    var ngramVocabSize: Int { ngramVocabSizeRatio * vocabSize }

    /// Number of n-gram embedders, `k * (n - 1)` in mlx-lm.
    var ngramEmbedderCount: Int { embSplitNum * (embNeighborNum - 1) }
}

extension SharpAILongcatFlashConfiguration: ModelConfigurationValidating {
    func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ reason: String) throws {
            if !condition {
                throw SharpAILongcatFlashError.invalidConfiguration(reason)
            }
        }

        try require(numLayers > 0, "num_layers must be positive")
        try require(numAttentionHeads > 0, "num_attention_heads must be positive")
        try require(nRoutedExperts > 0, "n_routed_experts must be positive")
        try require(zeroExpertNum >= 0, "zero_expert_num must not be negative")
        try require(
            moeTopk > 0 && moeTopk <= nRoutedExperts + zeroExpertNum,
            "moe_topk must be between 1 and n_routed_experts + zero_expert_num")
        try require(
            zeroExpertType == "identity", "only zero_expert_type \"identity\" is supported")
        try require(embNeighborNum >= 2, "emb_neighbor_num must be at least 2")
        try require(embSplitNum >= 1, "emb_split_num must be at least 1")
        try require(ngramVocabSizeRatio >= 1, "ngram_vocab_size_ratio must be at least 1")
        try require(
            hiddenSize % ngramEmbedderCount == 0,
            "hidden_size must be divisible by emb_split_num * (emb_neighbor_num - 1)")
    }
}
