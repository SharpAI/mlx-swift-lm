// SharpAI fork overlay: NemotronH_Nano_VL_V2 creator. Not part of upstream mlx-swift-lm.
//
// The tower and projector are in SharpAINemotronhNanoVlV2Vision.swift, and SharpAINemotronhNanoVlV2Model
// composes them with the text stack. The checkpoint loader is not wired (the sanitizer drops vision keys),
// so this creator still decodes and validates the config, then fails at load time rather than return
// a model with random vision weights.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAINemotronhNanoVlV2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "NemotronH_Nano_VL_V2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAINemotronhNanoVlV2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAINemotronhNanoVlV2Error.forwardNotImplemented(configuration.modelType)
        }
    ]
}
