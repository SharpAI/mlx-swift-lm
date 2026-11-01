// SharpAI fork overlay: cohere_compass creator. Not part of upstream mlx-swift-lm.
//
// The text decoder exists (SharpAICohereCompassTextModel), but the vision tower and the
// VL wrapper do not, so the creator still stops here (SharpAICohereCompassError.forwardNotImplemented).
// The creator decodes and validates the config, then fails with a clear error at load time.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAICohereCompassOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "cohere_compass": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAICohereCompassConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAICohereCompassError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
