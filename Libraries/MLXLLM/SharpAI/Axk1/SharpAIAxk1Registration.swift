// SharpAI fork overlay: axk1 creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as `LLMTypeRegistry`. LLMModelFactory is not edited; the overlay
// consumer merges these creators into its own type registry.

import Foundation
import MLXLMCommon

public enum SharpAIAxk1Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "axk1": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIAxk1Configuration.self, from: data)
            return SharpAIAxk1Model(configuration)
        }
    ]
}
