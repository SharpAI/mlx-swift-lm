// SharpAI fork overlay: qwen3_asr creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as VLMTypeRegistry: decode, validate, construct. VLMModelFactory is not
// edited. A consumer merges these creators into its registry. The ASR model and the forced
// aligner share the model_type, so the configuration picks between them.

import Foundation
import MLXLMCommon

public enum SharpAIQwen3AsrOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "qwen3_asr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIQwen3AsrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            if configuration.forcedAligner {
                return SharpAIQwen3AsrForcedAlignerForTokenClassification(configuration)
            }
            return SharpAIQwen3AsrForConditionalGeneration(configuration)
        }
    ]
}
