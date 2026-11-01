// SharpAI fork overlay: moondream3 creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory and
// VLMTypeRegistry are not edited, so the consumer merges `creators` itself. The processor is not
// registered in VLMProcessorTypeRegistry; use `makeProcessor` until it is.

import Foundation
import MLXLMCommon

public enum SharpAIMoondream3Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "moondream3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMoondream3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIMoondream3Model(configuration)
        }
    ]

    /// Builds the processor from the same `config.json` bytes the model creator reads.
    public static func makeProcessor(
        configurationData data: Data, tokenizer: any Tokenizer
    ) throws -> SharpAIMoondream3Processor {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMoondream3Configuration.self, from: data)
        try configuration.validateModelConfiguration()
        return SharpAIMoondream3Processor(configuration, tokenizer: tokenizer)
    }
}
