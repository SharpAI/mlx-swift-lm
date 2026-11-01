// SharpAI fork overlay: deepseek_v41 creator. Not part of upstream mlx-swift-lm.
//
// The creator decodes and validates the config, then builds the text model. It rejects a config
// without text_config, and Engram layers (mlx-lm's ModelArgs.__post_init__ rejects them too).
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
//
// Known gaps (not implemented): fp8/fp4 activation fake-quant, pipeline and tensor sharding,
// Engram, the DSpark drafter (DSparkV41DraftModel shares this model_type), the MTP head, the
// vision tower, and the fp8/fp4 checkpoint scale repack (see SharpAIDeepseekV41Sanitizer).

import Foundation
import MLXLMCommon

public enum SharpAIDeepseekV41Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "deepseek_v41": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDeepseekV41Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            guard let text = configuration.text else {
                throw SharpAIDeepseekV41Error.missingTextConfig
            }
            guard text.engramLayerIds.isEmpty else {
                throw SharpAIDeepseekV41Error.engramNotSupported(text.engramLayerIds)
            }
            return SharpAIDeepseekV41Model(text)
        }
    ]
}
