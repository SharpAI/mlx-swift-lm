// SharpAI fork overlay: muse_glimmer_assistant creator. Not part of upstream mlx-swift-lm.
//
// The drafter needs the Muse-Glimmer target's hidden states, and its forward pass is not
// implemented yet. The creator decodes and validates the config, then throws. Target
// compatibility is checked with validateTarget(_:) once a drafter instance exists.
// VLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIMuseGlimmerAssistantOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMModelFactory`'s registry.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "muse_glimmer_assistant": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMuseGlimmerAssistantConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMuseGlimmerAssistantError.drafterNotImplemented(configuration.modelType)
        }
    ]
}
