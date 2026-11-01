// SharpAI fork overlay: param2moe creator. Not part of upstream mlx-swift-lm.
//
// Decode, validate, then construct, as in LLMModelFactory. LLMModelFactory is not edited.
// The owner wires this overlay in.

import Foundation
import MLXLMCommon

public enum SharpAIParam2moeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "param2moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIParam2moeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIParam2moeModel(configuration)
        }
    ]
}
