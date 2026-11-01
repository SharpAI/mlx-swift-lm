// SharpAI fork overlay: phi4-siglip creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as `create` in VLMModelFactory. VLMModelFactory and
// VLMTypeRegistry are not edited, so the consumer merges `creators` itself. The processor is not
// registered in VLMProcessorTypeRegistry; use `makeProcessor` until it is.

import Foundation
import MLXLMCommon

public enum SharpAIPhi4SiglipOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "phi4-siglip": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPhi4SiglipConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIPhi4SiglipModel(configuration)
        }
    ]

    /// Builds the processor from the same `config.json` bytes the model creator reads.
    public static func makeProcessor(
        configurationData data: Data, tokenizer: any Tokenizer
    ) throws -> SharpAIPhi4SiglipProcessor {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIPhi4SiglipConfiguration.self, from: data)
        try configuration.validateModelConfiguration()
        return SharpAIPhi4SiglipProcessor(configuration, tokenizer: tokenizer)
    }
}
