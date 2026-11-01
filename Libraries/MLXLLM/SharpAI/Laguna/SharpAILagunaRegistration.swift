// SharpAI fork overlay: laguna creator. Not part of upstream mlx-swift-lm.
//
// Decode, validate, then construct, as in LLMModelFactory. LLMModelFactory is not edited.
// The owner wires this overlay in. The DFlash drafter is not registered here.

import Foundation
import MLXLMCommon

public enum SharpAILagunaOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "laguna": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAILagunaConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAILagunaModel(configuration)
        }
    ]
}
