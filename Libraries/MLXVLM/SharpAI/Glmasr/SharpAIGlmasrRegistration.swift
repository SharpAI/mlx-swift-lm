// SharpAI fork overlay: glmasr creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No processor is registered: the GLM-ASR prompt expansion of audio placeholders is not written.

import Foundation
import MLXLMCommon

public enum SharpAIGlmasrOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "glmasr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIGlmasrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIGlmasr(configuration)
        }
    ]
}
