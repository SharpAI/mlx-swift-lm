// SharpAI fork overlay: audioflamingo3 creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No processor is registered: the feature extractor and <sound> expansion are not written yet.

import Foundation
import MLXLMCommon

public enum SharpAIAudioflamingo3Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "audioflamingo3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIAudioflamingo3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIAudioflamingo3(configuration)
        }
    ]
}
