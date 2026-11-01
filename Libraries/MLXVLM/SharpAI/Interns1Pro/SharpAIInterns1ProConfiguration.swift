// SharpAI fork overlay: interns1_pro configuration. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/internlm/Intern-S1-Pro/resolve/main/config.json
// The text_config, the rope_scaling FoPE fields and the top-level tie_word_embeddings are decoded
// from that file. vision_config, ts_config and quantization_config are not decoded here: the
// vision tower and the time-series encoder are not implemented, and FP8 scales are handled by
// SharpAIInterns1ProSanitizer.

import Foundation
import MLXLMCommon

/// Text decoder configuration (`text_config`, model_type `interns1_pro_text`).
public struct SharpAIInterns1ProTextConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let hiddenAct: String

    /// MoE fields. `numExperts` is 0 for a checkpoint without routed experts.
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let decoderSparseStep: Int
    public let mlpOnlyLayers: [Int]
    public let normTopkProb: Bool
    /// Grouped router: experts are split into this many groups and `numExpertsPerToken /
    /// routerNGroups` experts are taken from each group. 1 selects the plain top-k router.
    public let routerNGroups: Int

    /// FoPE (Fourier position embedding) fields from `rope_scaling`. `useFope` follows the
    /// reference rule: any of fope_init_factor, fope_sep_head or num_inv_freq present.
    public let ropeType: String
    public let useFope: Bool
    public let fopeSepHead: Bool
    public let fopeInitFactor: Double?
    public let numInvFreq: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case hiddenAct = "hidden_act"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case normTopkProb = "norm_topk_prob"
        case routerNGroups = "router_n_groups"
        case ropeScaling = "rope_scaling"
    }

    /// Fields of `rope_scaling` that this port reads. Other keys are ignored.
    struct RopeScaling: Decodable {
        let ropeType: String?
        let fopeInitFactor: Double?
        let fopeSepHead: Bool?
        let numInvFreq: Int?

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case legacyType = "type"
            case fopeInitFactor = "fope_init_factor"
            case fopeSepHead = "fope_sep_head"
            case numInvFreq = "num_inv_freq"
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            // `rope_type` is the current key; `type` is the older spelling.
            ropeType =
                try c.decodeIfPresent(String.self, forKey: .ropeType)
                ?? c.decodeIfPresent(String.self, forKey: .legacyType)
            fopeInitFactor = try c.decodeIfPresent(Double.self, forKey: .fopeInitFactor)
            fopeSepHead = try c.decodeIfPresent(Bool.self, forKey: .fopeSepHead)
            numInvFreq = try c.decodeIfPresent(Int.self, forKey: .numInvFreq)
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / max(attentionHeads, 1))
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"

        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 0
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? intermediateSize
        decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        routerNGroups = try c.decodeIfPresent(Int.self, forKey: .routerNGroups) ?? 1

        let scaling = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        ropeType = scaling?.ropeType ?? "default"
        fopeInitFactor = scaling?.fopeInitFactor
        fopeSepHead = scaling?.fopeSepHead ?? false
        numInvFreq = scaling?.numInvFreq
        useFope =
            scaling.map {
                $0.fopeInitFactor != nil || $0.fopeSepHead != nil || $0.numInvFreq != nil
            } ?? false
    }

    /// Rejects shapes this port cannot run. Throws `ModelFactoryError.invalidConfiguration`.
    public func validate() throws {
        func fail(_ message: String) -> ModelFactoryError {
            ModelFactoryError.invalidConfiguration("interns1_pro text_config: \(message)")
        }
        guard hiddenAct == "silu" else {
            throw fail("hidden_act must be silu, got \(hiddenAct)")
        }
        guard ropeType == "default" else {
            throw fail("rope_type \(ropeType) is not supported (only default)")
        }
        guard numInvFreq == nil else {
            throw fail("num_inv_freq is not supported (FoPE frequencies must be head_dim/2)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0 else {
            throw fail("hidden_size, num_hidden_layers and vocab_size must be positive")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw fail("num_attention_heads must be a positive multiple of num_key_value_heads")
        }
        guard headDim > 0, headDim % 2 == 0 else {
            throw fail("head_dim must be a positive even number")
        }
        guard decoderSparseStep > 0 else {
            throw fail("decoder_sparse_step must be positive")
        }
        if numExperts > 0 {
            guard numExpertsPerToken > 0, numExpertsPerToken <= numExperts else {
                throw fail("num_experts_per_tok must be in 1...num_experts")
            }
            guard routerNGroups >= 1, numExperts % routerNGroups == 0 else {
                throw fail("num_experts must be divisible by router_n_groups")
            }
            guard numExpertsPerToken % routerNGroups == 0 else {
                throw fail("num_experts_per_tok must be divisible by router_n_groups")
            }
        }
    }
}

public struct SharpAIInterns1ProConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "interns1_pro"

    public let modelType: String
    public let architectures: [String]
    /// Absent in a config that only names the model type. The creator requires it.
    public let textConfig: SharpAIInterns1ProTextConfiguration?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfig = try c.decodeIfPresent(SharpAIInterns1ProTextConfiguration.self, forKey: .textConfig)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "interns1_pro: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        try textConfig?.validate()
    }
}
