// SharpAI fork overlay: dots3_note creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet. This creator decodes and validates the config, then fails
// with a clear error at load time. Wiring into VLMTypeRegistry is left to the overlay consumer;
// VLMModelFactory is not edited. dots3_note_mtp is a separate type and is not registered here.

import Foundation
import MLXLMCommon

public enum SharpAIDots3NoteOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "dots3_note": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDots3NoteConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIDots3NoteError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
