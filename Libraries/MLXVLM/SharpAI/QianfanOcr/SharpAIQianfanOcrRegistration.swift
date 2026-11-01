// SharpAI fork overlay: qianfan_ocr creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate and construct sequence as VLMTypeRegistry. VLMModelFactory is not edited.
// Wiring into the registry is left to the overlay consumer. No processor is registered.

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
}
