// SharpAI fork overlay: moss_transcribe_diarize creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as VLMModelFactory. VLMModelFactory is not edited.
// No processor is registered: the audio placeholder expansion is not written.

import Foundation
import MLXLMCommon

public enum SharpAIMossTranscribeDiarizeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "moss_transcribe_diarize": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMossTranscribeDiarizeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIMossTranscribeDiarize(configuration)
        }
    ]
}
