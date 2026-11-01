// SharpAI fork overlay: flint creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape and decode/validate/construct sequence as `LLMTypeRegistry`.
// LLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIFlintOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "flint": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIFlintConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIFlintModel(configuration)
        }
    ]
}
