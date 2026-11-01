// SharpAI fork overlay: step3p7 creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.
// The construct step throws. The step3p5 text backbone is SharpAIStep3p7TextModel (MLXLLM overlay).
// The PE vision tower and projector are not implemented, so no multimodal model is built yet.

import Foundation
import MLXLMCommon

public enum SharpAIStep3p7Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step3p7": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStep3p7Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIStep3p7Error.unsupported(
                "perception_encoder vision tower and projector are not implemented")
        }
    ]
}
