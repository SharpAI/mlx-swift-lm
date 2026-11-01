// SharpAI fork overlay: minimax_m3_vl configuration. Not part of upstream mlx-swift-lm.
//
// Only top-level keys are typed. text_config and vision_config stay raw because no M3
// config source exists in this repo, vLLM, or mlx-lm on this machine.
// UNVERIFIED: key names and the architecture string come from the model spec, not a checkpoint.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3VlError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "MinimaxM3Vl unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "MinimaxM3Vl invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: minimax_m3_vl`.
public struct SharpAIMinimaxM3VlConfiguration: Codable, Sendable {
    public static let expectedArchitecture = "MiniMaxM3SparseForConditionalGeneration"

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
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3_vl"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfig = try container.decode(JSONValue.self, forKey: .textConfig)
        visionConfig = try container.decodeIfPresent(JSONValue.self, forKey: .visionConfig)
    }

    /// Checks the model type, the architecture name when present, and that text_config is an object.
    public func validateModelConfiguration() throws {
        guard modelType == "minimax_m3_vl" else {
            throw SharpAIMinimaxM3VlError.invalidConfiguration("model_type must be minimax_m3_vl")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIMinimaxM3VlError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard case .object = textConfig else {
            throw SharpAIMinimaxM3VlError.invalidConfiguration("text_config must be an object")
        }
    }
}
