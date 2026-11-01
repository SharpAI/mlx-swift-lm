// SharpAI fork overlay: muse_spark configuration. Not part of upstream mlx-swift-lm.
//
// Only the identity fields are decoded. The checkpoint config is gated, so the text and
// vision blocks are not decoded yet.
// UNVERIFIED: the architecture name `MuseSparkForConditionalGeneration` comes from the model spec only.

import Foundation
import MLXLMCommon

public enum SharpAIMuseSparkError: LocalizedError {
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "Model type muse_spark is registered, but the model is not implemented: \(detail)"
        }
    }
}

public struct SharpAIMuseSparkConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "muse_spark"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    public func validateModelConfiguration() throws {
        guard modelType == "muse_spark" else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_spark: unexpected model_type \(modelType)")
        }
    }
}
