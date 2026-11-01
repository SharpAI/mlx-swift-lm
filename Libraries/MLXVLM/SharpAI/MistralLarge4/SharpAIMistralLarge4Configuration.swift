// SharpAI fork overlay: mistral_large4 configuration. Not part of upstream mlx-swift-lm.
//
// Only top-level keys are typed. text_config and vision_config stay raw because no Mistral
// Large 4 config source exists in this repo, vLLM, or mlx-lm on this machine.
// UNVERIFIED: key names and the architecture string come from the model spec, not a checkpoint.

import Foundation
import MLXLMCommon

public enum SharpAIMistralLarge4Error: LocalizedError {
    case forwardNotImplemented(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(architecture UNVERIFIED)."
        case .invalidConfiguration(let detail):
            return "MistralLarge4 invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: mistral_large4`.
public struct SharpAIMistralLarge4Configuration: Codable, Sendable {
    public static let expectedModelType = "mistral_large4"
    public static let expectedArchitecture = "MistralLarge4ForConditionalGeneration"

    public let modelType: String
    public let architectures: [String]
    public let textConfig: JSONValue
    public let visionConfig: JSONValue?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfig = try container.decode(JSONValue.self, forKey: .textConfig)
        visionConfig = try container.decodeIfPresent(JSONValue.self, forKey: .visionConfig)
    }

    /// Checks the model type, the architecture name when present, and that text_config is an object.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIMistralLarge4Error.invalidConfiguration(
                "model_type must be \(Self.expectedModelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIMistralLarge4Error.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard case .object = textConfig else {
            throw SharpAIMistralLarge4Error.invalidConfiguration(
                "text_config must be an object")
        }
    }

    /// True when text_config sets tie_word_embeddings to true. False when absent.
    public var tiedWordEmbeddings: Bool {
        guard case .object(let text) = textConfig,
            case .bool(let tied)? = text["tie_word_embeddings"]
        else {
            return false
        }
        return tied
    }
}
