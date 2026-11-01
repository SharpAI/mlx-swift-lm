// SharpAI fork overlay: hunyuan_vl creator. Not part of upstream mlx-swift-lm.
//
// The text decoder and vision tower exist (SharpAIHunyuanVlModel.swift, random-weight tested).
// This creator stays gated: SharpAIHunyuanVlModel is not a LanguageModel, has no KV cache wiring
// and no checkpoint key mapping, so it cannot generate. It decodes and validates the config, then
// fails. VLMModelFactory is not edited. Wiring is left to the overlay consumer.

import Foundation
import MLXLMCommon

public enum SharpAIHunyuanVlOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "hunyuan_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIHunyuanVlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIHunyuanVlError.forwardNotImplemented(configuration.modelType)
        }
    ]
}
