// SharpAI fork overlay: qwen3_omni_moe creator. Not part of upstream mlx-swift-lm.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No processor is registered: the feature extractor and audio/image placeholder expansion
// are not written here.

import Foundation
import MLXLMCommon

public enum SharpAIQwen3OmniMoeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) static let thinkerCreator: (Data) throws -> any LanguageModel = { data in
        let configuration = try JSONDecoder.json5().decode(
            SharpAIQwen3OmniMoeConfiguration.self, from: data)
        try configuration.validateModelConfiguration()
        return try SharpAIQwen3OmniMoeThinker(configuration)
    }

    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "qwen3_omni_moe": thinkerCreator,
        // UNVERIFIED: alias for a config that drops the "_moe" suffix. Same decode path.
        "qwen3_omni": thinkerCreator,
    ]
}
