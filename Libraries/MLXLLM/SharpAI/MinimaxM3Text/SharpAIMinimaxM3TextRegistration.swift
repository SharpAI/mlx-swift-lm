// SharpAI fork overlay: minimax_m3_text creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is not
// edited; the overlay consumer merges these creators. Construction builds
// SharpAIMinimaxM3TextModel: per-head qk norm, dense and sparse-MoE layers, and the block indexer.
// The checkpoint layout is UNVERIFIED (see SharpAIMinimaxM3TextModel.swift).

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3TextOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m3_text": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM3TextConfiguration.self, from: data)
            try configuration.validateForConstruction()
            return SharpAIMinimaxM3TextModel(configuration)
        }
    ]
}
