// SharpAI fork overlay: olmo_hybrid creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as the `create` helper in LLMModelFactory.swift.
// Wiring into LLMTypeRegistry is left to the overlay consumer; LLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIOlmoHybridOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "olmo_hybrid": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOlmoHybridConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIOlmoHybridModel(configuration)
        }
    ]
}
