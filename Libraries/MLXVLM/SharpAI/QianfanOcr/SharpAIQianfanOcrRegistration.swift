// SharpAI fork overlay: qianfan_ocr creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate and construct sequence as VLMTypeRegistry. VLMModelFactory is not edited.
// The aggregate in SharpAIVLMModelOverlays wires it into VLMTypeRegistry, and its processor
// (SharpAIQianfanOcrProcessor) is built from the same config.json bytes through `makeProcessor`.

import Foundation
import MLXLMCommon

public enum SharpAIQianfanOcrOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "qianfan_ocr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIQianfanOcrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIQianfanOcr(configuration)
        }
    ]

    /// Builds the processor from the same `config.json` bytes the model creator reads.
    public static func makeProcessor(
        configurationData data: Data, tokenizer: any Tokenizer
    ) throws -> SharpAIQianfanOcrProcessor {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIQianfanOcrConfiguration.self, from: data)
        try configuration.validateModelConfiguration()
        return SharpAIQianfanOcrProcessor(configuration, tokenizer: tokenizer)
    }
}
