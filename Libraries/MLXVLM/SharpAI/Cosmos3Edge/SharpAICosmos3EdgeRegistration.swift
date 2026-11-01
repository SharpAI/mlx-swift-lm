// SharpAI fork overlay: cosmos3_edge creator. Not part of upstream mlx-swift-lm.
//
// The text decoder exists (SharpAICosmos3EdgeTextForCausalLM, in SharpAICosmos3EdgeLanguage.swift),
// but the full VLM class is not written. This creator still decodes and validates the config, then
// fails with a clear error at load time, so the registry never hands out an unloadable model.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
//
// Known gaps (not implemented):
// - Vision tower (SigLIP2-style encoder with per-frame attention) and projector (patch merger).
// - Processor for image and video inputs.
// - Checkpoint key mapping for the language tower (see the sanitizer file).

import Foundation
import MLXLMCommon

public enum SharpAICosmos3EdgeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "cosmos3_edge": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAICosmos3EdgeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAICosmos3EdgeError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
