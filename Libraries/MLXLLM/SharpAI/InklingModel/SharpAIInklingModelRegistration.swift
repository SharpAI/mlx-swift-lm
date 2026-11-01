// SharpAI fork overlay: inkling_model creator. Not part of upstream mlx-swift-lm.
//
// Decodes the config, then builds the text model. Wiring into LLMTypeRegistry is left to the
// overlay consumer; LLMModelFactory is not edited.
// `inkling_mm_model` is the published multimodal config. Its text backbone loads here and its
// vision and audio weights are dropped by the sanitizer.

import Foundation
import MLXLMCommon

public enum SharpAIInklingModelOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "inkling_model": { data in try SharpAIInklingModelOverlay.makeModel(data) },
        "inkling_mm_model": { data in try SharpAIInklingModelOverlay.makeModel(data) },
    ]

    static func makeModel(_ data: Data) throws -> any LanguageModel {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIInklingModelConfiguration.self, from: data)
        return SharpAIInklingModel(configuration.text)
    }
}
