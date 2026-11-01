// SharpAI fork overlay: granite_speech_plus creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.

import Foundation
import MLXLMCommon

public enum SharpAIGraniteSpeechPlusOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "granite_speech_plus": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIGraniteSpeechPlusConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIGraniteSpeechPlusForConditionalGeneration(configuration)
        }
    ]
}
