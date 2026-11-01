// SharpAI fork overlay: ovis2_6_moe creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet: the MoE text decoder, visual tokenizer and visual embedding
// table have no reference source in the repo. This creator decodes and validates the config, then fails.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIOvis26MoeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "ovis2_6_moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOvis26MoeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIOvis26MoeError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
