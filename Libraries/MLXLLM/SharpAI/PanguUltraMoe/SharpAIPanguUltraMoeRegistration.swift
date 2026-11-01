// SharpAI fork overlay: pangu_ultra_moe creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as `LLMTypeRegistry`: decode, validate, construct. LLMModelFactory is
// not edited; the overlay consumer merges these creators into its own registry.

import Foundation
import MLXLMCommon

public enum SharpAIPanguUltraMoeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "pangu_ultra_moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPanguUltraMoeConfiguration.self, from: data)
            try configuration.validateForForward()
            return SharpAIPanguUltraMoeModel(configuration)
        }
    ]
}
