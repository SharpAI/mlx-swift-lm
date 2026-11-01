// SharpAI fork overlay: kanana-1.5-v creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory is not
// edited, so the consumer merges `creators` itself. No processor is registered yet.

import Foundation
import MLXLMCommon

public enum SharpAIKanana15VOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "kanana-1.5-v": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIKanana15VConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIKanana15VModel(configuration)
        }
    ]
}
