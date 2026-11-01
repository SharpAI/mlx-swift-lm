// SharpAI fork overlay: granite4_vision creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet: the Q-Former projector and vision tower have no reference
// source in the repo. This creator decodes and validates the config, then fails at load time.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIGranite4VisionOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "granite4_vision": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIGranite4VisionConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIGranite4VisionError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
