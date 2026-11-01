// SharpAI fork overlay: minimax_m2 configuration. Not part of upstream mlx-swift-lm.
//
// The spec entry for minimax_m2 is the EAGLE3 drafter (arch Eagle3MiniMaxM2ForCausalLM), not the
// base model. Keys follow the standard HF draft config.
//
// Verified against: https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3/resolve/main/config.json
// (fetched as a summary, not verbatim). It lists hidden_size 3072, intermediate_size 8192,
// num_hidden_layers 1, num_attention_heads 24, num_key_value_heads 8, vocab_size 200064,
// draft_vocab_size 32000, rms_norm_eps 1e-6, rope_theta 5000000, max_position_embeddings 196608.
// The public drafter declares model_type "llama" and architectures ["LlamaForCausalLMEagle3"], not
// minimax_m2 and Eagle3MiniMaxM2ForCausalLM, so validateModelConfiguration() rejects the real
// drafter config. That mismatch is open (see the KNOWN GAP in the registration tests).
// The default values for optional keys are still a guess and are not source-verified.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM2Error: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case blockedOnBase

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason):
            return "Invalid minimax_m2 configuration: \(reason)"
        case .blockedOnBase:
            return
                "Model type minimax_m2 is not loadable yet: the EAGLE3 drafter needs the MiniMax M2 "
                + "target model, which is not implemented (UNVERIFIED weight layout)."
        }
    }
}

public struct SharpAIMinimaxM2Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "minimax_m2"

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int?
    public let vocabSize: Int
    public let draftVocabSize: Int?
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int?

    /// Attention head width. Falls back to hidden size divided by head count.
    public var resolvedHeadDim: Int {
        headDim ?? (hiddenSize / max(numAttentionHeads, 1))
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case draftVocabSize = "draft_vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let heads = try container.decode(Int.self, forKey: .numAttentionHeads)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 1
        numAttentionHeads = heads
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? heads
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        draftVocabSize = try container.decodeIfPresent(Int.self, forKey: .draftVocabSize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIMinimaxM2Error.invalidConfiguration(
                "model_type is \(modelType), expected \(Self.expectedModelType)")
        }
        guard hiddenSize > 0, numHiddenLayers > 0, vocabSize > 0 else {
            throw SharpAIMinimaxM2Error.invalidConfiguration(
                "hidden_size, num_hidden_layers and vocab_size must be positive")
        }
        guard numAttentionHeads > 0, numKeyValueHeads > 0,
            numAttentionHeads % numKeyValueHeads == 0
        else {
            throw SharpAIMinimaxM2Error.invalidConfiguration(
                "num_key_value_heads must be positive and divide num_attention_heads")
        }
        guard resolvedHeadDim > 0 else {
            throw SharpAIMinimaxM2Error.invalidConfiguration("head dimension must be positive")
        }
        if let draftVocabSize, draftVocabSize <= 0 || draftVocabSize > vocabSize {
            throw SharpAIMinimaxM2Error.invalidConfiguration(
                "draft_vocab_size must be in 1...vocab_size")
        }
    }
}
