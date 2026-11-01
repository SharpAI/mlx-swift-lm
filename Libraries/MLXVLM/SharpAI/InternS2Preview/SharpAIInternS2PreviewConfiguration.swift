// SharpAI fork overlay: intern_s2_preview configuration. Not part of upstream mlx-swift-lm.
//
// The text and vision fields use the upstream Qwen35Configuration. This type only checks model_type.

import Foundation
import MLXLMCommon

public struct SharpAIInternS2PreviewConfiguration:
    Decodable, Sendable, ModelConfigurationValidating
{
    public static let expectedModelType = "intern_s2_preview"

    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "intern_s2_preview: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
    }
}
