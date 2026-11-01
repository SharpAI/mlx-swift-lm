// SharpAI fork overlay: isaac creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, the same sequence as VLMTypeRegistry, then builds
// SharpAIIsaacModel. Weight loading goes through SharpAIIsaacModel.sanitize(weights:).
// VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIIsaacOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "isaac": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIIsaacConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIIsaacModel(configuration)
        }
    ]
}
