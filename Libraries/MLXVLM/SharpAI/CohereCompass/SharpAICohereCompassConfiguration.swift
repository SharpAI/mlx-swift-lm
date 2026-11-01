// SharpAI fork overlay: cohere_compass configuration. Not part of upstream mlx-swift-lm.
//
// Only `model_type` and `architectures` are decoded here. The text sub-config is decoded by
// SharpAICohereCompassTextConfiguration. The vision sub-config is not modeled yet.
// Reference: transformers modeling_cohere_compass.py (PR #47878); checkpoint config in
// scripts/conformance/offline/fixtures/cohere_compass.

import Foundation
import MLXLMCommon

public enum SharpAICohereCompassError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its model class is not implemented yet "
                + "(architecture UNVERIFIED)."
        }
    }
}

public struct SharpAICohereCompassConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// Architecture name from the model spec. Checked only when `architectures` is present.
    public static let expectedArchitecture = "CohereCompassForConditionalGeneration"

    public let modelType: String
    public let architectures: [String]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere_compass"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures)
    }

    public func validateModelConfiguration() throws {
        guard let architectures else { return }
        guard architectures.contains(Self.expectedArchitecture) else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere_compass: architectures must include \(Self.expectedArchitecture)")
        }
    }
}
