// SharpAI fork overlay: minimax_m3_mtp configuration. Not part of upstream mlx-swift-lm.
//
// Only the keys the spec names are typed. Everything else stays in the raw JSON.
// Verified against: https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/models/minimax_m3/amd/mtp.py
//   (the MTP head reads `num_mtp_modules` for its layer count and `hidden_size`, `rms_norm_eps`).
// Verified against: https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json
//   (`num_mtp_modules` = 7 and `num_nextn_predict_layers` = 1 both sit under `text_config`).
// UNVERIFIED: the standalone `minimax_m3_mtp` config layout (flat or nested) and the architecture
// string "MiniMaxM3MTP". The public M3 config has no such file, and "MiniMaxM3MTP" is the vLLM class
// name, not a config `architectures` value.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3MtpError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "MinimaxM3Mtp unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "MinimaxM3Mtp invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: minimax_m3_mtp`.
public struct SharpAIMinimaxM3MtpConfiguration: Codable, Sendable {
    public static let expectedModelType = "minimax_m3_mtp"
    public static let expectedArchitecture = "MiniMaxM3MTP"

    public let modelType: String
    public let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    /// Checks the model type and, when present, the architecture name.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
    }
}

/// MTP depth count read from the same `config.json`.
///
/// vLLM builds `num_mtp_modules` MTP layers (`MiniMaxM3MultiTokenPredictor`). The public M3
/// `text_config` also lists `num_nextn_predict_layers` = 1, which disagrees with 7. This decoder
/// follows vLLM: `num_mtp_modules` first, then `num_nextn_predict_layers`, then 1.
public struct SharpAIMinimaxM3MtpDepthConfiguration: Decodable, Sendable {
    public let numMtpModules: Int

    enum CodingKeys: String, CodingKey {
        case numMtpModules = "num_mtp_modules"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        numMtpModules =
            try container.decodeIfPresent(Int.self, forKey: .numMtpModules)
            ?? container.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers)
            ?? 1
    }
}
