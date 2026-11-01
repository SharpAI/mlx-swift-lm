// SharpAI fork overlay: funaudiochat creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No model class exists yet, so the creator decodes and validates the config, then throws.

import Foundation
import MLXLMCommon

public enum SharpAIFunaudiochatOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "funaudiochat": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIFunaudiochatConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIFunaudiochatError.unsupported(
                "model class not written; audio towers and Qwen3 wiring not ported yet")
        }
    ]
}
