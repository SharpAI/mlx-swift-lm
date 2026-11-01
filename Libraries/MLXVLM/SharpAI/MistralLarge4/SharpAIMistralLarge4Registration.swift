// SharpAI fork overlay: mistral_large4 creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as `VLMTypeRegistry`: decode, validate, construct. VLMModelFactory is
// not edited; the overlay consumer merges these creators into its own registry.
// The construct step throws: no model class exists yet, and the architecture is UNVERIFIED.
// Gap: a checkpoint whose outer model_type is "mistral3" with text_config.model_type
// "mistral_large4" never reaches this key. The consumer must route it here.

import Foundation
import MLXLMCommon

public enum SharpAIMistralLarge4Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "mistral_large4": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMistralLarge4Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMistralLarge4Error.forwardNotImplemented(
                configuration.modelType)
        }
    ]
}
