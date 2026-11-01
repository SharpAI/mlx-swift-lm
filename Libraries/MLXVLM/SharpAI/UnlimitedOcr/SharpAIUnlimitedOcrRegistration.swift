// SharpAI fork overlay: unlimited_ocr creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAIUnlimitedOcrModel (text decoder, plus the
// SAM and CLIP-L vision stack when the config has vision_config).
// VLMModelFactory is not edited; the overlay consumer wires these creators in.

import Foundation
import MLXLMCommon

public enum SharpAIUnlimitedOcrOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "unlimited_ocr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIUnlimitedOcrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIUnlimitedOcrModel(configuration)
        }
    ]
}
