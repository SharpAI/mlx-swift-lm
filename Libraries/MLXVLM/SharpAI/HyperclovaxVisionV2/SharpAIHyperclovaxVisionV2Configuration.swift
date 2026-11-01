// SharpAI fork overlay: hyperclovax_vision_v2 configuration. Not part of upstream mlx-swift-lm.
//
// Text and vision blocks are decoded as dimensions only. The repo has no text model_type for
// hyperclovax, and the ViT used by the reference vision stack is fileprivate, so neither has a model class.
// The text_config / vision_config nesting matches the conformance fixture config. No checkpoint was read.

import Foundation
import MLXLMCommon

public enum SharpAIHyperclovaxVisionV2Error: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(text decoder and ViT missing)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after hyperclovax_vision_v2 sanitize: \(key)"
        }
    }
}

/// Text dimensions read from the text block.
public struct SharpAIHyperclovaxVisionV2TextDimensions: Decodable, Sendable {
    public let modelType: String?
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
    }
}

public struct SharpAIHyperclovaxVisionV2Configuration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textDimensions: SharpAIHyperclovaxVisionV2TextDimensions
    public let hasVisionConfig: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "hyperclovax_vision_v2"

        textDimensions = try container.decode(
            SharpAIHyperclovaxVisionV2TextDimensions.self, forKey: .textConfig)

        hasVisionConfig = container.contains(.visionConfig)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.vocabSize, text.hiddenSize, text.numHiddenLayers,
            text.numAttentionHeads, text.numKeyValueHeads,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "hyperclovax_vision_v2: text vocab, hidden, layer and head sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "hyperclovax_vision_v2: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
