// SharpAI fork overlay: k2_horizon creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as `LLMTypeRegistry`. LLMModelFactory is not edited; the overlay
// consumer merges these creators into its own type registry.

import Foundation
import MLXLMCommon

public enum SharpAIK2HorizonOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "k2_horizon": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIK2HorizonConfiguration.self, from: data)
            return SharpAIK2HorizonModel(configuration)
        }
    ]
}
