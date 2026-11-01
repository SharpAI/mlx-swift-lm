// SharpAI fork overlay: qwen3_5_moe_text creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is not
// edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIQwen35MoeTextOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "qwen3_5_moe_text": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIQwen35MoeTextConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIQwen35MoeTextModel(configuration.textConfiguration)
        }
    ]
}
