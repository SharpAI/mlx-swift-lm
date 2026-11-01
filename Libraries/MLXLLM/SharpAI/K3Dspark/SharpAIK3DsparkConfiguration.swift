// SharpAI fork overlay: k3_dspark configuration. Not part of upstream mlx-swift-lm.
//
// Field names and checks follow vLLM's K3DSparkConfig (vllm/transformers_utils/configs/k3_dspark.py).
// Defaults for keys the fixture omits follow DeepseekV3Config. UNVERIFIED: rms_norm_eps and
// max_position_embeddings defaults, because the fixture sets both.

import Foundation
import MLXLMCommon

public enum SharpAIK3DsparkError: LocalizedError {
    case targetNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .targetNotImplemented(let type):
            return
                "Model type \(type) is a DSpark drafter. Its target model (Kimi K3) is not "
                + "implemented, so the drafter cannot load on its own."
        }
    }
}

public struct SharpAIK3DsparkConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "k3_dspark"

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    /// MLA query low-rank. nil is rejected by validation.
    public let qLoraRank: Int?
    public let kvLoraRank: Int
    public let qkNopeHeadDim: Int
    public let qkRopeHeadDim: Int
    public let vHeadDim: Int
    public let vocabSize: Int
    public let draftVocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    /// From `rope_parameters`, or `rope_scaling` when that is absent.
    public let ropeScaling: [String: StringOrNumber]?
    public let numTargetLayers: Int
    public let targetHiddenSize: Int
    public let targetLayerIds: [Int]
    public let markovRank: Int
    public let enableConfidenceHead: Bool
    public let confidenceHeadWithMarkov: Bool
    public let mlaUseNope: Bool
    public let mlaUseOutputGate: Bool
    public let mlaUseQkNorm: Bool
    public let dsparkBonusAnchor: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case vocabSize = "vocab_size"
        case draftVocabSize = "draft_vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case targetHiddenSize = "target_hidden_size"
        case targetLayerIds = "target_layer_ids"
        case markovRank = "markov_rank"
        case enableConfidenceHead = "enable_confidence_head"
        case confidenceHeadWithMarkov = "confidence_head_with_markov"
        case mlaUseNope = "mla_use_nope"
        case mlaUseOutputGate = "mla_use_output_gate"
        case mlaUseQkNorm = "mla_use_qk_norm"
        case dsparkBonusAnchor = "dspark_bonus_anchor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        draftVocabSize = try c.decodeIfPresent(Int.self, forKey: .draftVocabSize) ?? vocabSize
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
        var theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 50000
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let ropeScalingValue = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        var scaling = ropeParameters ?? ropeScalingValue
        // rope_theta nested in rope_parameters takes precedence over the top-level value.
        if let nestedTheta = scaling?["rope_theta"]?.asFloat() {
            theta = nestedTheta
            scaling?["rope_theta"] = nil
        }
        ropeTheta = theta
        ropeScaling = scaling
        numTargetLayers = try c.decode(Int.self, forKey: .numTargetLayers)
        targetHiddenSize = try c.decode(Int.self, forKey: .targetHiddenSize)
        targetLayerIds = try c.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        markovRank = try c.decode(Int.self, forKey: .markovRank)
        enableConfidenceHead =
            try c.decodeIfPresent(Bool.self, forKey: .enableConfidenceHead) ?? false
        confidenceHeadWithMarkov =
            try c.decodeIfPresent(Bool.self, forKey: .confidenceHeadWithMarkov) ?? false
        mlaUseNope = try c.decodeIfPresent(Bool.self, forKey: .mlaUseNope) ?? false
        mlaUseOutputGate = try c.decodeIfPresent(Bool.self, forKey: .mlaUseOutputGate) ?? false
        mlaUseQkNorm = try c.decodeIfPresent(Bool.self, forKey: .mlaUseQkNorm) ?? false
        dsparkBonusAnchor = try c.decodeIfPresent(Bool.self, forKey: .dsparkBonusAnchor) ?? false
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "k3_dspark: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        var unsupported: [String] = []
        if mlaUseNope { unsupported.append("mla_use_nope") }
        if mlaUseOutputGate { unsupported.append("mla_use_output_gate") }
        if mlaUseQkNorm { unsupported.append("mla_use_qk_norm") }
        if dsparkBonusAnchor { unsupported.append("dspark_bonus_anchor") }
        if qLoraRank == nil { unsupported.append("q_lora_rank=None") }
        if !unsupported.isEmpty {
            throw ModelFactoryError.invalidConfiguration(
                "MLA DSpark does not support " + unsupported.joined(separator: ", ") + ".")
        }

        guard draftVocabSize == vocabSize else {
            throw ModelFactoryError.invalidConfiguration(
                "MLA DSpark requires draft_vocab_size to equal vocab_size when sharing the "
                + "target embedding and LM head.")
        }

        guard !targetLayerIds.isEmpty, numTargetLayers == targetLayerIds.count else {
            throw ModelFactoryError.invalidConfiguration(
                "MLA DSpark requires non-empty target_layer_ids and a matching num_target_layers.")
        }
    }
}
