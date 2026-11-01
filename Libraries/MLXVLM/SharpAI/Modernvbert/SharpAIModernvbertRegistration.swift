// SharpAI fork overlay: modernvbert creator. Not part of upstream mlx-swift-lm.
//
// Returns an encoder for retrieval, not a generator. VLMTypeRegistry is not edited, so the
// consumer merges `SharpAIModernvbertOverlay.creators` when it needs this model type.

import Foundation
import MLXLMCommon

public enum SharpAIModernvbertOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "modernvbert": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIModernvbertConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIModernvbert(configuration)
        }
    ]
}
