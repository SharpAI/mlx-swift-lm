// SharpAI fork overlay: cohere_asr creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.

import Foundation
import MLXLMCommon

public enum SharpAICohereAsrOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "cohere_asr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAICohereAsrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAICohereAsrForConditionalGeneration(configuration)
        }
    ]
}
