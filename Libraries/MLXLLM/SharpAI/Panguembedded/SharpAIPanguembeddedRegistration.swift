// SharpAI fork overlay: PanguEmbedded creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry. LLMModelFactory is not edited. The overlay
// consumer merges these creators into its own registry.

import Foundation
import MLXLMCommon

public enum SharpAIPanguembeddedOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "PanguEmbedded": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPanguembeddedConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIPanguembeddedModel(configuration)
        }
    ]
}
