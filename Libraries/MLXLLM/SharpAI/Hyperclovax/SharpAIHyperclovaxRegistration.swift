// SharpAI fork overlay: hyperclovax creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape and decode, validate, construct order as LLMTypeRegistry.
// LLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIHyperclovaxOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hyperclovax": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHyperclovaxConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIHyperclovaxModel(configuration)
        }
    ]
}
