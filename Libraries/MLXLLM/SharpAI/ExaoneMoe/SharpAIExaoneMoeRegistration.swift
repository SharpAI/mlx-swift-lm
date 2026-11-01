// SharpAI fork overlay: exaone_moe creator. Not part of upstream mlx-swift-lm.
//
// Decode, validate, then construct, as in LLMModelFactory. LLMModelFactory is not edited.
// The owner wires this overlay in. The MTP predictor is not registered here (no drafter
// protocol conformance yet).

import Foundation
import MLXLMCommon

public enum SharpAIExaoneMoeOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "exaone_moe": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIExaoneMoeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIExaoneMoeModel(configuration)
        }
    ]
}
