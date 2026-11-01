// SharpAI fork overlay: qwen3_5_moe_text configuration. Not part of upstream mlx-swift-lm.
//
// A flat text-only config (no `text_config` wrapper). The fields come from the upstream
// `Qwen35TextConfiguration`, which already reads the MoE keys (`num_experts`, `moe_intermediate_size`).

import Foundation
import MLXLMCommon

public enum SharpAIQwen35MoeTextError: LocalizedError, Equatable {
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Qwen35MoeText invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: qwen3_5_moe_text`.
public struct SharpAIQwen35MoeTextConfiguration: Decodable, Sendable {
    public static let expectedModelType = "qwen3_5_moe_text"
    public static let expectedArchitecture = "Qwen3_5MoeForCausalLM"

    public let modelType: String
    public let architectures: [String]
    public let textConfiguration: Qwen35TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfiguration = try Qwen35TextConfiguration(from: decoder)
    }

    /// Checks the model type, the architecture name when present, and the MoE shape.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIQwen35MoeTextError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIQwen35MoeTextError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        let text = textConfiguration
        guard text.numExperts > 0, text.moeIntermediateSize > 0 else {
            throw SharpAIQwen35MoeTextError.invalidConfiguration(
                "num_experts and moe_intermediate_size must be positive")
        }
        guard text.numExpertsPerTok > 0, text.numExpertsPerTok <= text.numExperts else {
            throw SharpAIQwen35MoeTextError.invalidConfiguration(
                "num_experts_per_tok must be between 1 and num_experts")
        }
    }
}
