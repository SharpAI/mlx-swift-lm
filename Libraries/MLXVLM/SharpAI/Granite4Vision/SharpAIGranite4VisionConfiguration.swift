// SharpAI fork overlay: granite4_vision configuration. Not part of upstream mlx-swift-lm.
//
// The text stack is a nested `text_config` block, decoded with the existing GraniteConfiguration.
// UNVERIFIED: the `text_config` nesting and the Granite text fields are assumed from the
// Granite 4.0 layout. The vision tower and Q-Former fields are not decoded: no reference source.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIGranite4VisionError: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(Q-Former projector and vision tower UNVERIFIED)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after granite4_vision sanitize: \(key)"
        }
    }
}

public struct SharpAIGranite4VisionConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textConfiguration: GraniteConfiguration
    public let hasVisionConfig: Bool

    private let textDimensions: TextDimensions

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    private struct TextDimensions: Decodable, Sendable {
        let hiddenSize: Int
        let hiddenLayers: Int
        let attentionHeads: Int
        let kvHeads: Int
        let vocabularySize: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "granite4_vision"

        if container.contains(.textConfig) {
            textConfiguration = try container.decode(
                GraniteConfiguration.self, forKey: .textConfig)
            textDimensions = try container.decode(TextDimensions.self, forKey: .textConfig)
        } else {
            textConfiguration = try GraniteConfiguration(from: decoder)
            textDimensions = try TextDimensions(from: decoder)
        }

        hasVisionConfig = container.contains(.visionConfig)
    }

    public func validateModelConfiguration() throws {
        let dims = textDimensions
        let values = [
            dims.hiddenSize, dims.hiddenLayers, dims.attentionHeads, dims.kvHeads,
            dims.vocabularySize,
        ]
        guard values.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "granite4_vision: text hidden, layer, head and vocab sizes must be positive")
        }
        guard dims.attentionHeads % dims.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "granite4_vision: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
