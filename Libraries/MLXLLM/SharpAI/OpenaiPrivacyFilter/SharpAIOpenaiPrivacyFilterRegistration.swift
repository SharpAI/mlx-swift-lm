// SharpAI fork overlay: openai_privacy_filter creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape and decode, validate, construct sequence as LLMTypeRegistry.
// LLMModelFactory is not edited. The overlay consumer merges these creators into its registry.

import Foundation
import MLXLMCommon

public enum SharpAIOpenaiPrivacyFilterOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "openai_privacy_filter": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOpenaiPrivacyFilterConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIOpenaiPrivacyFilterModel(configuration)
        }
    ]
}
