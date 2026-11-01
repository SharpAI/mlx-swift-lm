// SharpAI fork overlay: paddleocr_vl creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No processor is registered: the PaddleOCR-VL image processor is not written.

import Foundation
import MLXLMCommon

public enum SharpAIPaddleocrVlOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "paddleocr_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPaddleocrVlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIPaddleocrVl(configuration)
        }
    ]
}
