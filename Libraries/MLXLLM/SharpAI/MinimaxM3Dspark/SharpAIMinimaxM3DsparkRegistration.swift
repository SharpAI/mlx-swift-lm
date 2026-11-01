// SharpAI fork overlay: minimax_m3_dspark creator. Not part of upstream mlx-swift-lm.
//
// A DSpark drafter needs its frozen MiniMax M3 target for embeddings, LM head and hidden states.
// That target is not implemented here, and the draft checkpoint is not public. The creator
// decodes and validates the config, then throws. LLMModelFactory is not edited; the overlay
// consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3DsparkOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m3_dspark": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM3DsparkConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMinimaxM3DsparkError.unsupported(
                "the draft checkpoint is not public and the MiniMax M3 target is not implemented")
        }
    ]
}
