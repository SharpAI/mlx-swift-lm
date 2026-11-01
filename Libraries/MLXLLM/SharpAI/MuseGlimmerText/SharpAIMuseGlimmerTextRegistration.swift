// SharpAI fork overlay: muse_glimmer_text creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape and decode, validate, construct order as `LLMTypeRegistry`. LLMModelFactory
// is not edited; the overlay consumer merges these creators into its own type registry.

import Foundation
import MLXLMCommon

public enum SharpAIMuseGlimmerTextOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "muse_glimmer_text": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMuseGlimmerTextConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIMuseGlimmerTextModel(configuration)
        }
    ]
}
