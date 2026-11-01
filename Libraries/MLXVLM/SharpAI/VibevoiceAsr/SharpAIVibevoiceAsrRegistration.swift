// SharpAI fork overlay: vibevoice_asr creator. Not part of upstream mlx-swift-lm.
//
// Same creator shape as VLMTypeRegistry: decode, validate, construct. VLMModelFactory is not
// edited. A consumer merges these creators into its registry.

import Foundation
import MLXLMCommon

public enum SharpAIVibevoiceAsrOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "vibevoice_asr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIVibevoiceAsrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIVibevoiceAsrForConditionalGeneration(configuration)
        }
    ]
}
