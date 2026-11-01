// SharpAI fork overlay: "unknown" model_type creator. Not part of upstream mlx-swift-lm.
//
// Both architectures share the placeholder key "unknown", so the creator picks the config by
// the `architectures` list. Same closure shape as `LLMTypeRegistry`. LLMModelFactory is not
// edited. Both paths throw after validation: the EAGLE draft is not a LanguageModel, and the
// PanguProMoEV2 forward pass is not written yet. The real config model_type is UNVERIFIED.

import Foundation
import MLXLMCommon

public enum SharpAIUnknownOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "unknown": { data in
            let architectures = try JSONDecoder.json5().decode(
                SharpAIUnknownArchitectures.self, from: data
            ).architectures

            if architectures.contains(SharpAIEagleCohereConfiguration.expectedArchitecture) {
                let configuration = try JSONDecoder.json5().decode(
                    SharpAIEagleCohereConfiguration.self, from: data)
                try configuration.validateModelConfiguration()
                throw SharpAIUnknownError.draftOnly(
                    SharpAIEagleCohereConfiguration.expectedArchitecture)
            }

            if architectures.contains(SharpAIPanguProMoEV2Configuration.expectedArchitecture) {
                let configuration = try JSONDecoder.json5().decode(
                    SharpAIPanguProMoEV2Configuration.self, from: data)
                try configuration.validateModelConfiguration()
                throw SharpAIUnknownError.forwardNotImplemented(
                    SharpAIPanguProMoEV2Configuration.expectedArchitecture)
            }

            throw SharpAIUnknownError.unsupportedArchitecture(architectures.joined(separator: ", "))
        }
    ]
}
