// SharpAI fork overlay: inkling_mm_model creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds the model: the text backbone plus the audio and
// vision towers. The text path runs a full-sequence forward. Incremental decoding is not implemented
// and throws from `newCache`. VLMModelFactory is not edited. The owner wires this into VLMTypeRegistry.

import Foundation
import MLXLMCommon

public enum SharpAIInklingMmModelError: LocalizedError {
    case incrementalDecodeNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .incrementalDecodeNotImplemented(let type):
            return
                "Model type \(type) runs full-sequence forward only. KV-cache and short-conv "
                + "state for incremental decoding is not implemented yet."
        }
    }
}

public enum SharpAIInklingMmModelOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "inkling_mm_model": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIInklingMmModelConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIInklingMmModel(configuration)
        }
    ]
}
