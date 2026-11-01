// SharpAI fork overlay: diffusion_gemma creator. Not part of upstream mlx-swift-lm.
//
// The block-diffusion denoise engine and the model class are not written: no reference source
// for the sampler is in the repo. The creator decodes and validates the config, then fails.
// VLMModelFactory is not edited. Wiring is left to the overlay consumer.

import Foundation
import MLXLMCommon

public enum SharpAIDiffusionGemmaOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "diffusion_gemma": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDiffusionGemmaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            // UNVERIFIED: no model class exists and this overlay is not wired into VLMTypeRegistry, so loading always fails here.
            throw SharpAIDiffusionGemmaError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
