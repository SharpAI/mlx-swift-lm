// SharpAI fork overlay: cohere2_moe creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAICohere2MoeModel. LLMModelFactory is not
// edited. The owner wires `creators` into LLMTypeRegistry.

import Foundation
import MLXLMCommon

public enum SharpAICohere2MoeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "cohere2_moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAICohere2MoeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAICohere2MoeModel(configuration)
        }
    ]
}
