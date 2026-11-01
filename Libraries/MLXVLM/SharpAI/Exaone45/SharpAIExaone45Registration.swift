// SharpAI fork overlay: exaone4_5 creator. Not part of upstream mlx-swift-lm.
//
// The vision tower and the forward pass are not written: no reference source for them is in the
// repo. The creator decodes and validates the config, checks the text config maps to
// Exaone4Configuration, then fails. VLMModelFactory is not edited. Wiring is left to the overlay
// consumer.

import Foundation
import MLXLMCommon

public enum SharpAIExaone45Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "exaone4_5": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIExaone45Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            _ = try configuration.exaone4TextConfiguration()
            throw SharpAIExaone45Error.forwardNotImplemented(configuration.modelType)
        }
    ]
}
