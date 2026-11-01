// SharpAI fork overlay: molmo2 creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet: the vision tower, image pooling and language stack have no
// reference source in the repo or on this machine. This creator decodes and validates the config,
// then fails at load time. Wiring into VLMTypeRegistry is left to the overlay consumer;
// VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIMolmo2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "molmo2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMolmo2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMolmo2Error.forwardNotImplemented(configuration.modelType)
        }
    ]
}
