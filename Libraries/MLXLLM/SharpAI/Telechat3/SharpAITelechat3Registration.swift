// SharpAI fork overlay: telechat3 creator. Not part of upstream mlx-swift-lm.
//
// Alias: telechat3 is built on the Llama layout, so its creator returns SharpAITelechat3Model,
// a LlamaModel subclass. Same closure shape as LLMTypeRegistry. LLMModelFactory is not edited.
// The overlay consumer merges these creators into its own registry.

import Foundation
import MLXLMCommon

public enum SharpAITelechat3Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "telechat3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAITelechat3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAITelechat3Model(configuration.llamaConfiguration)
        }
    ]
}
