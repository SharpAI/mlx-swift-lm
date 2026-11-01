// SharpAI fork overlay: intern_s2_preview creator. Not part of upstream mlx-swift-lm.
//
// Uses the same decode, validate and construct sequence as VLMTypeRegistry. VLMModelFactory is
// not edited. Wiring into the registry is left to the overlay consumer.

import Foundation
import MLXLMCommon

public enum SharpAIInternS2PreviewOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "intern_s2_preview": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIInternS2PreviewConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            let qwenConfiguration = try JSONDecoder.json5().decode(
                Qwen35Configuration.self, from: data)
            return SharpAIInternS2PreviewModel(qwenConfiguration)
        }
    ]
}
