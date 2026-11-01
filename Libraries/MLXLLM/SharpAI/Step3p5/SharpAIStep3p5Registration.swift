// SharpAI fork overlay: step3p5 creator. Not part of upstream mlx-swift-lm.
//
// Decode, validate, then construct, as in LLMModelFactory. LLMModelFactory is not edited.
// The owner wires this overlay in. The Step3p5MTP draft head is not registered here.

import Foundation
import MLXLMCommon

public enum SharpAIStep3p5Overlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step3p5": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStep3p5Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIStep3p5Model(configuration)
        }
    ]
}
