// SharpAI fork overlay: minimax_m3_vl creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.
// The construct step throws: the vision tower and the block-sparse attention are not
// implemented, and no M3 model source exists in this repo, vLLM, or mlx-lm on this machine.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3VlOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m3_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM3VlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMinimaxM3VlError.unsupported(
                "vision tower and block-sparse attention are not implemented")
        }
    ]
}
