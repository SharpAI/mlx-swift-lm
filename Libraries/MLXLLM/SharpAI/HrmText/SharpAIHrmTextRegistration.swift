// SharpAI fork overlay: hrm_text creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, then construct the model.
// Wiring into LLMTypeRegistry is left to the overlay consumer; LLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIHrmTextOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hrm_text": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHrmTextConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIHrmTextForCausalLM(configuration)
        }
    ]
}
