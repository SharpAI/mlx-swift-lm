// SharpAI fork overlay: UltraVBERT creator. Not part of upstream mlx-swift-lm.
//
// The model class is not written yet, so the creator decodes the config and then throws.
// VLMTypeRegistry is not edited. The consumer merges `SharpAIUltravbertOverlay.creators`.

import Foundation
import MLXLMCommon

public enum SharpAIUltravbertError: Error, Equatable {
    /// The config parsed, but the encoder for this model type is not implemented.
    case notImplemented(modelType: String)
}

public enum SharpAIUltravbertOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "ultravbert": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIUltravbertConfiguration.self, from: data)
            throw SharpAIUltravbertError.notImplemented(modelType: configuration.modelType)
        }
    ]
}
