// SharpAI fork overlay: minimax_m2 creator. Not part of upstream mlx-swift-lm.
//
// minimax_m2 is an EAGLE3 drafter for the MiniMax M2 base model, which is not implemented here.
// The creator decodes and validates the config, then fails with a clear error and builds no model.
// LLMModelFactory is not edited; the overlay consumer wires this in.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM2Overlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m2": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM2Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMinimaxM2Error.blockedOnBase
        }
    ]
}
