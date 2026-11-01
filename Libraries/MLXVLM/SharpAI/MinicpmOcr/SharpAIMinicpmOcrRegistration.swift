// SharpAI fork overlay: minicpm_ocr creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No slice processor is registered: the MiniCPM-OCR image slicing is not written, so callers
// must supply already-sliced pixels.

import Foundation
import MLXLMCommon

public enum SharpAIMinicpmOcrOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minicpm_ocr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinicpmOcrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIMinicpmOcr(configuration)
        }
    ]
}
