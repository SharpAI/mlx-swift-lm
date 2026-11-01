// SharpAI fork overlay: kimi_k3 configuration. Not part of upstream mlx-swift-lm.
//
// Top-level keys are typed here. text_config is kept raw and decoded by SharpAIKimiK3TextConfiguration.
// Verified against: https://huggingface.co/moonshotai/Kimi-K3/resolve/main/config.json (model_type kimi_k3,
// architectures KimiK3ForConditionalGeneration, text_config, vision_config).

import Foundation
import MLXLMCommon

public enum SharpAIKimiK3Error: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "KimiK3 unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "KimiK3 invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: kimi_k3`.
public struct SharpAIKimiK3Configuration: Codable, Sendable {
    public static let expectedArchitecture = "KimiK3ForConditionalGeneration"

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
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "kimi_k3"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfig = try container.decode(JSONValue.self, forKey: .textConfig)
        visionConfig = try container.decodeIfPresent(JSONValue.self, forKey: .visionConfig)
    }

    /// Checks the model type, the architecture name when present, and that text_config is an object.
    public func validateModelConfiguration() throws {
        guard modelType == "kimi_k3" else {
            throw SharpAIKimiK3Error.invalidConfiguration("model_type must be kimi_k3")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIKimiK3Error.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard case .object = textConfig else {
            throw SharpAIKimiK3Error.invalidConfiguration("text_config must be an object")
        }
    }
}
