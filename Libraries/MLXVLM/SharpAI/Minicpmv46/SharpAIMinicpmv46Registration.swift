// SharpAI fork overlay: minicpmv4_6 creator. Not part of upstream mlx-swift-lm.
//
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
// No processor is registered: the MiniCPM-V slice processor is not written.

import Foundation
import MLXLMCommon

public enum SharpAIMinicpmv46Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minicpmv4_6": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinicpmv46Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIMinicpmv46(configuration)
        }
    ]
}
