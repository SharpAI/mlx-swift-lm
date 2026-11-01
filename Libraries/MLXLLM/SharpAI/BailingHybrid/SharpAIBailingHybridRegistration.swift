// SharpAI fork overlay: bailing_hybrid creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAIBailingHybridModel. Configs that lack the
// MLA or grouping fields fail with ModelFactoryError.invalidConfiguration before any weights are built.
// Wiring into LLMTypeRegistry is left to the overlay consumer; LLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIBailingHybridOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    /// Not wired into LLMTypeRegistry by this overlay.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "bailing_hybrid": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIBailingHybridConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIBailingHybridModel(configuration)
        }
    ]
}
