// SharpAI fork overlay: molmo2 configuration. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the `text_config` nesting and its field names are assumed from the common Hugging
// Face layout. The `vit_config` block is only detected, not decoded: no reference source.

import Foundation
import MLXLMCommon

public enum SharpAIMolmo2Error: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(vision tower and language stack UNVERIFIED)."
        }
    }
}

public struct SharpAIMolmo2Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hasVisionConfig: Bool

    private let textDimensions: TextDimensions

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vit_config"
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
            ?? "molmo2"
        textDimensions = try container.decode(TextDimensions.self, forKey: .textConfig)
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
                "molmo2: text hidden, layer, head and vocab sizes must be positive")
        }
        guard dims.attentionHeads % dims.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "molmo2: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
