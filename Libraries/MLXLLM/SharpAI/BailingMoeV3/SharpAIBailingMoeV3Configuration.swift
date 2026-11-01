//
//  SharpAIBailingMoeV3Configuration.swift
//  MLXLLM
//
//  Text config for `bailing_moe_v3` (Ling-3.0). Field names follow the
//  checkpoint config.json, as read by vLLM's bailing_moe_v3.py.
//  UNVERIFIED: no real checkpoint config was loaded; values come from the
//  VL fixture's text_config and the vLLM source.
//

import Foundation
import MLXLMCommon

enum SharpAIBailingMoeV3Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "bailing_moe_v3 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "bailing_moe_v3 invalid configuration: \(message)"
        }
    }
}

struct SharpAIBailingMoeV3Configuration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var maxPositionEmbeddings: Int?

    // MLA dimensions.
    var qkNopeHeadDim: Int
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qLoraRank: Int?
    var kvLoraRank: Int

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

    // Layout and optional features.
    var tieWordEmbeddings: Bool
    var useBias: Bool
    var layerGroupSize: Int
    var shortConvKernelSize: Int
    var gatedAttentionProjGranularityType: String?
    var expertSwigluLimitList: [Double]?
    var shareExpertSwigluLimitList: [Double]?
    /// [T, H, W] sections of the M-RoPE on the MLA rope slice.
    var mropeSection: [Int]?

    // KDA (linear attention) layers.
    /// Per-head width of the KDA projections (`head_dim`).
    var headDim: Int
    /// `no_kda_lora`: true means full-rank f_proj / g_proj. False (low-rank) is not implemented.
    var noKdaLora: Bool
    /// `kda_safe_gate`: bounded gate lower_bound * sigmoid(...) instead of -exp(A) * softplus(...).
    var kdaSafeGate: Bool
    /// `kda_lower_bound`: lower bound of the log-decay when kdaSafeGate is set.
    var kdaLowerBound: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBias = "use_bias"
        case layerGroupSize = "layer_group_size"
        case shortConvKernelSize = "short_conv_kernel_size"
        case gatedAttentionProjGranularityType = "gated_attention_proj_granularity_type"
        case expertSwigluLimitList = "expert_swiglu_limit_list"
        case shareExpertSwigluLimitList = "share_expert_swiglu_limit_list"
        case mropeSection = "mrope_section"
        case headDim = "head_dim"
        case noKdaLora = "no_kda_lora"
        case kdaSafeGate = "kda_safe_gate"
        case kdaLowerBound = "kda_lower_bound"
    }

    /// Decode-only key for `use_kda_lora`. It has no stored property, so it must stay out of
    /// `CodingKeys`: a CodingKey without a matching stored property breaks synthesized `Encodable`.
    private enum LegacyCodingKeys: String, CodingKey {
        case useKdaLora = "use_kda_lora"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "bailing_moe_v3"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)

        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512

        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        numSharedExperts = try c.decode(Int.self, forKey: .numSharedExperts)
        moeSharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
        nGroup = try c.decode(Int.self, forKey: .nGroup)
        topkGroup = try c.decode(Int.self, forKey: .topkGroup)
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0

        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useBias = try c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
        layerGroupSize = try c.decode(Int.self, forKey: .layerGroupSize)
        shortConvKernelSize = try c.decodeIfPresent(Int.self, forKey: .shortConvKernelSize) ?? 4
        gatedAttentionProjGranularityType = try c.decodeIfPresent(
            String.self, forKey: .gatedAttentionProjGranularityType)
        expertSwigluLimitList = try c.decodeIfPresent([Double].self, forKey: .expertSwigluLimitList)
        shareExpertSwigluLimitList = try c.decodeIfPresent(
            [Double].self, forKey: .shareExpertSwigluLimitList)
        mropeSection = try c.decodeIfPresent([Int].self, forKey: .mropeSection)

        // Defaults to hidden_size / num_attention_heads when the checkpoint omits head_dim
        // (UNVERIFIED: default convention taken from the usual HF pattern).
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        // The tiny checkpoint sets both no_kda_lora and use_kda_lora; no_kda_lora wins when present.
        let legacy = try decoder.container(keyedBy: LegacyCodingKeys.self)
        let useKdaLora = try legacy.decodeIfPresent(Bool.self, forKey: .useKdaLora) ?? false
        noKdaLora = try c.decodeIfPresent(Bool.self, forKey: .noKdaLora) ?? !useKdaLora
        kdaSafeGate = try c.decodeIfPresent(Bool.self, forKey: .kdaSafeGate) ?? true
        kdaLowerBound = try c.decodeIfPresent(Float.self, forKey: .kdaLowerBound) ?? -5.0
    }
}

extension SharpAIBailingMoeV3Configuration: ModelConfigurationValidating {

    /// True when layer `index` uses KDA linear attention (vLLM `_is_kda_layer`).
    func isKDALayer(_ index: Int) -> Bool {
        let groupEnd = (hiddenLayers / layerGroupSize) * layerGroupSize
        return !((index + 1) % layerGroupSize == 0 || index >= groupEnd)
    }

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        guard layerGroupSize > 0 else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration("layer_group_size must be > 0")
        }
        guard numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration(
                "num_experts_per_tok must be in 1...num_experts")
        }
        guard nGroup > 0, numExperts % nGroup == 0, topkGroup > 0, topkGroup <= nGroup else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration(
                "n_group and topk_group must divide num_experts")
        }
        guard qkRopeHeadDim > 0, qkRopeHeadDim % 2 == 0 else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration("qk_rope_head_dim must be even")
        }
        if let section = mropeSection {
            guard section.count == 3, section[1] == section[2],
                section[0] + section[1] + section[2] == qkRopeHeadDim / 2
            else {
                throw SharpAIBailingMoeV3Error.invalidConfiguration(
                    "mrope_section must be [T, H, H] summing to qk_rope_head_dim / 2")
            }
        }
        if let granularity = gatedAttentionProjGranularityType, granularity != "head_wise" {
            throw SharpAIBailingMoeV3Error.unsupported(
                "gated_attention_proj_granularity_type=\(granularity)")
        }
        if qLoraRank != nil {
            throw SharpAIBailingMoeV3Error.unsupported("q_lora_rank (MLA query LoRA) is not implemented")
        }
        guard headDim > 0 else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration("head_dim must be > 0")
        }
        guard shortConvKernelSize >= 1 else {
            throw SharpAIBailingMoeV3Error.invalidConfiguration("short_conv_kernel_size must be >= 1")
        }
        if !noKdaLora, (0 ..< hiddenLayers).contains(where: { isKDALayer($0) }) {
            throw SharpAIBailingMoeV3Error.unsupported(
                "low-rank KDA projections (no_kda_lora=false) are not implemented")
        }
        let clamps = (expertSwigluLimitList ?? []) + (shareExpertSwigluLimitList ?? [])
        if clamps.contains(where: { $0 != 0 }) {
            throw SharpAIBailingMoeV3Error.unsupported(
                "per-layer SwiGLU clamp (expert_swiglu_limit_list) is not implemented")
        }
    }
}
