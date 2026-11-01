// SharpAI fork overlay: moss_audio creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as VLMModelFactory. VLMModelFactory is not edited.
// Wiring into VLMTypeRegistry is left to the overlay consumer.
// No processor is registered: the audio prompt expansion and time markers are not written.

import Foundation
import MLXLMCommon

public enum SharpAIMossAudioOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "moss_audio": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMossAudioConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIMossAudio(configuration)
        }
    ]
}
