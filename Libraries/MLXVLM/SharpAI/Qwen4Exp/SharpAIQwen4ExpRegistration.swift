// SharpAI fork overlay: qwen4_exp creator. Not part of upstream mlx-swift-lm.
//
// Builds the text path of Qwen4ExpForCausalLM (model.language_model.* plus lm_head). The vision
// tower (model.visual.*) and the MTP head (mtp.*) are not ported; the sanitizer drops their keys.
// VLMTypeRegistry is not edited; the overlay consumer wires this dictionary in.

import Foundation
import MLXLMCommon

// SharpAIQwen4ExpError is declared in SharpAIQwen4ExpConfiguration.swift.

public enum SharpAIQwen4ExpOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "qwen4_exp": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIQwen4ExpConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIQwen4ExpForCausalLM(configuration)
        }
    ]
}
