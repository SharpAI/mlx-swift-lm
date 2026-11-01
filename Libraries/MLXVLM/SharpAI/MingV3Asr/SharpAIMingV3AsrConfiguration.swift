// SharpAI fork overlay: ming_v3_asr configuration. Not part of upstream mlx-swift-lm.
//
// Only the top-level `model_type` is decoded. The audio encoder (whisper_enc) and the text
// backbone (KDA + MLA + MoE, the bailing_moe_v3 family) are not written yet, so validation
// always throws `.unsupported` after the model type check.
// UNVERIFIED: the Ming-3.0-asr config field names are not on disk. No config.json was read.

import Foundation
import MLXLMCommon

public enum SharpAIMingV3AsrError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let reason):
            return "MingV3Asr: unsupported: \(reason)."
        case .invalidConfiguration(let reason):
            return "MingV3Asr: invalid configuration: \(reason)."
        }
    }
}

public struct SharpAIMingV3AsrConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
    }

    public func validateModelConfiguration() throws {
        guard modelType == "ming_v3_asr" else {
            throw SharpAIMingV3AsrError.invalidConfiguration(
                "model_type must be ming_v3_asr, got \(modelType)")
        }
        throw SharpAIMingV3AsrError.unsupported(
            "audio encoder (whisper_enc) and text backbone (KDA/MLA/MoE, bailing_moe_v3) are not implemented"
        )
    }
}
