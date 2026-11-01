// SharpAI fork overlay: k3_dspark creator. Not part of upstream mlx-swift-lm.
//
// A DSpark drafter has no standalone LanguageModel. It needs the frozen Kimi K3 target for
// embeddings, LM head and target hidden states, and that target is not implemented. The creator
// decodes and validates the config, then throws. SharpAIK3DsparkDraftModel is the class to
// attach once the target exists. LLMModelFactory is not edited; the overlay consumer merges
// these creators.

import Foundation
import MLXLMCommon

public enum SharpAIK3DsparkOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "k3_dspark": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIK3DsparkConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIK3DsparkError.targetNotImplemented(configuration.modelType)
        }
    ]
}
