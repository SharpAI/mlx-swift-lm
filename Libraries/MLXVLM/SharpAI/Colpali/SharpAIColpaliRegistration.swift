// SharpAI fork overlay: colpali creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAIColpali. Wiring into VLMTypeRegistry
// is left to the overlay consumer; VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIColpaliOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "colpali": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIColpaliConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIColpali(configuration)
        }
    ]
}
