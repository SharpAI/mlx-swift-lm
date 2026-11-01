// SharpAI fork overlay: llava_onevision2 creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config with the same sequence as VLMTypeRegistry, then builds
// SharpAILlavaOnevision2Model. Weight loading goes through SharpAILlavaOnevision2Model.sanitize(weights:).
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAILlavaOnevision2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "llava_onevision2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAILlavaOnevision2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAILlavaOnevision2Model(configuration)
        }
    ]
}
