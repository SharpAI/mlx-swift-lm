// SharpAI fork overlay: kolibri1 configuration. Not part of upstream mlx-swift-lm.
//
// Field names and values come from the Aleph-Alpha/Kolibri-1 config fixture
// (scripts/conformance/offline/fixtures/kolibri1). Only the fixture fields are decoded.

import Foundation
import MLXLMCommon

public enum SharpAIKolibri1Error: Error, LocalizedError, Equatable {
    case invalidConfiguration(String)
    case modelClassNotImplemented

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason):
            return "Invalid kolibri1 configuration: \(reason)"
        case .modelClassNotImplemented:
            return "kolibri1 model class is not implemented yet. No modeling source was available."
        }
    }
}

public struct SharpAIKolibri1Configuration: Decodable, Sendable {
    public let modelType: String
    public let architectures: [String]
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool
    public let useSlidingWindow: Bool
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let normTopkProb: Bool
    public let eosTokenId: Int?
    public let padTokenId: Int?

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "kolibri1"
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try container.decode(Float.self, forKey: .ropeTheta)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        numExperts = try container.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try container.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        normTopkProb = try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId)
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
    }

    /// True when the layer uses sliding-window attention. Out-of-range indices return false.
    public func isSlidingAttentionLayer(_ index: Int) -> Bool {
        layerTypes.indices.contains(index) && layerTypes[index] == "sliding_attention"
    }
}

extension SharpAIKolibri1Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard numKeyValueHeads > 0, numAttentionHeads % numKeyValueHeads == 0 else {
            throw SharpAIKolibri1Error.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard layerTypes.count == numHiddenLayers else {
            throw SharpAIKolibri1Error.invalidConfiguration(
                "layer_types has \(layerTypes.count) entries, expected \(numHiddenLayers)")
        }
        guard numExperts > 0, (1 ... numExperts).contains(numExpertsPerTok) else {
            throw SharpAIKolibri1Error.invalidConfiguration(
                "num_experts_per_tok must be between 1 and num_experts")
        }
    }
}
