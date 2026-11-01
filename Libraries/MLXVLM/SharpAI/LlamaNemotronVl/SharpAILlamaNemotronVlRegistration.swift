// SharpAI fork overlay: llama_nemotron_vl creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAILlamaNemotronVl. Wiring into VLMTypeRegistry
// is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAILlamaNemotronVlOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "llama_nemotron_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAILlamaNemotronVlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAILlamaNemotronVl(configuration)
        }
    ]
}
