// SharpAI fork overlay: minimax_m3_dspark configuration. Not part of upstream mlx-swift-lm.
//
// Only model_type and architectures are typed. The draft checkpoint is not public (HF 404),
// and no DSpark or MiniMax M3 draft source exists in this repo, vLLM, or mlx-lm on this machine.
// UNVERIFIED: the architecture name comes from the model spec, not from a checkpoint config.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3DsparkError: LocalizedError {
    case invalidConfiguration(String)
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "MinimaxM3Dspark invalid configuration: \(detail)"
        case .unsupported(let detail):
            return "MinimaxM3Dspark unsupported: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: minimax_m3_dspark`.
public struct SharpAIMinimaxM3DsparkConfiguration: Decodable, Sendable {
    public static let expectedModelType = "minimax_m3_dspark"
    public static let expectedArchitecture = "DSparkMiniMaxDraftModel"

    public let modelType: String
    public let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    /// Checks the model type, and the architecture name when the config lists architectures.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIMinimaxM3DsparkError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIMinimaxM3DsparkError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
    }
}
