// SharpAI fork overlay: alpamayo2_super creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet (see SharpAIAlpamayo2SuperError.forwardNotImplemented).
// This creator decodes and validates the config, then fails with a clear error at load time.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIAlpamayo2SuperOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "alpamayo2_super": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIAlpamayo2SuperConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIAlpamayo2SuperError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
