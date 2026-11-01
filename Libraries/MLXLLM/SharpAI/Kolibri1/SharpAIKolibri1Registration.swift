// SharpAI fork overlay: kolibri1 creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as `LLMTypeRegistry`. LLMModelFactory is not edited; the overlay
// consumer merges these creators into its own type registry.
// The creator validates the config and builds `SharpAIKolibri1Model`.

import Foundation
import MLXLMCommon

public enum SharpAIKolibri1Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "kolibri1": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIKolibri1Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIKolibri1Model(configuration)
        }
    ]
}
