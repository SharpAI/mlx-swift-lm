// SharpAI fork overlay: Dory creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry. LLMModelFactory is not edited.
// The config decodes, validates, and builds SharpAIDoryForCausalLM with the cache plan.
// Weights are not loaded here. The checkpoint loader applies them after creation.

import Foundation
import MLXLMCommon

public enum SharpAIDoryOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "dory": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDoryConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIDoryForCausalLM(configuration)
        }
    ]
}
