// SharpAI fork overlay: params.json creator. Not part of upstream mlx-swift-lm.
//
// params.json is the Mistral native config. Its model is the EAGLE drafter
// EagleMistralLarge3ForCausalLM. The creator decodes and validates the config, then refuses to
// build a standalone LanguageModel, because the drafter needs target hidden states.
// LLMModelFactory is not edited; the overlay consumer wires this in.

import Foundation
import MLXLMCommon

public enum SharpAIParamsJsonOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "params.json": { data in
            let configuration = try SharpAIParamsJsonConfiguration.decode(data)
            try configuration.validate()
            throw SharpAIParamsJsonError.eagleRequiresTargetModel("params.json")
        }
    ]
}
