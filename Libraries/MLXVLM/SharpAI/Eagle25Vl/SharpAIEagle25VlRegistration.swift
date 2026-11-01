// SharpAI fork overlay: eagle_2_5_vl creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory and
// VLMTypeRegistry are not edited, so the consumer merges `creators` itself. The processor is not
// registered in VLMProcessorTypeRegistry; use `makeProcessor` until it is.

import Foundation
import MLXLMCommon

public enum SharpAIEagle25VlOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "eagle_2_5_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIEagle25VlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIEagle25VlModel(configuration)
        }
    ]

    /// Builds the processor from the same `config.json` bytes the model creator reads.
    public static func makeProcessor(
        configurationData data: Data, tokenizer: any Tokenizer
    ) throws -> SharpAIEagle25VlProcessor {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIEagle25VlConfiguration.self, from: data)
        try configuration.validateModelConfiguration()
        return SharpAIEagle25VlProcessor(configuration, tokenizer: tokenizer)
    }
}
