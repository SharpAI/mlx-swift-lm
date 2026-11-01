// SharpAI fork overlay: ming_v3_asr creator. Not part of upstream mlx-swift-lm.
//
// Same decode, validate, construct sequence as VLMModelFactory. The model class is not written
// yet, so the creator fails after validation. VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

public enum SharpAIMingV3AsrOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "ming_v3_asr": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMingV3AsrConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMingV3AsrError.unsupported("MingV3Asr model class is not written yet")
        }
    ]
}
