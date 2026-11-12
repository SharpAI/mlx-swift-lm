// SharpAI fork overlay: nemotron_parse creator. Not part of upstream mlx-swift-lm.
//
// Builds the RADIO encoder, the mBART-style decoder, and the untied lm_head. VLMTypeRegistry and
// VLMModelFactory are not edited. The aggregate in SharpAIVLMModelOverlays wires the creator into
// VLMTypeRegistry and registers SharpAINemotronParseProcessor under its `processor_class`.

import Foundation
import MLXLMCommon

public enum SharpAINemotronParseOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "nemotron_parse": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAINemotronParseConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAINemotronParse(configuration)
        }
    ]

    /// Builds the processor from the checkpoint's `preprocessor_config.json` bytes.
    public static func makeProcessor(
        configurationData data: Data, tokenizer: any Tokenizer
    ) throws -> SharpAINemotronParseProcessor {
        let configuration = try JSONDecoder.json5().decode(
            SharpAINemotronParseImageConfiguration.self, from: data)
        return SharpAINemotronParseProcessor(configuration, tokenizer: tokenizer)
    }
}
