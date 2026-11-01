// SharpAI fork overlay: pangu_ultra_moe_mtp configuration. Not part of upstream mlx-swift-lm.
//
// Only the keys the spec names are typed. Everything else in config.json is ignored.
// Key names `num_hidden_layers` and `num_nextn_predict_layers` follow vLLM's openpangu_mtp.py
// (`OpenPanguMultiTokenPredictor` reads `config.num_hidden_layers` and
// `config.num_nextn_predict_layers`). The MTP layer sits at index `num_hidden_layers` in the
// checkpoint. The published openPangu-Ultra-MoE-718B config.json carries model_type
// `pangu_ultra_moe` with `num_hidden_layers` 61 and `num_nextn_predict_layers` 1.
// Verified against: https://huggingface.co/FreedomIntelligence/openPangu-Ultra-MoE-718B/raw/main/config.json
// No config.json uses the `pangu_ultra_moe_mtp` name; the overlay accepts it as the spec names it.
// The base dimensions for the MTP block come from the same JSON, decoded by
// `SharpAIPanguUltraMoeConfiguration` (see `baseConfiguration(from:)`).

import Foundation
import MLXLMCommon

public enum SharpAIPanguUltraMoeMtpError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "PanguUltraMoeMtp unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "PanguUltraMoeMtp invalid configuration: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: pangu_ultra_moe_mtp`.
public struct SharpAIPanguUltraMoeMtpConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "pangu_ultra_moe_mtp"
    public static let expectedArchitecture = "OpenPanguMTPModel"

    public let modelType: String
    public let architectures: [String]
    /// Main-model decoder layer count. The MTP layer index starts here. Nil when absent.
    public let hiddenLayers: Int?
    /// Number of next-token-prediction layers. Nil when absent.
    public let nextnPredictLayers: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenLayers = "num_hidden_layers"
        case nextnPredictLayers = "num_nextn_predict_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
        nextnPredictLayers =
            try container.decodeIfPresent(Int.self, forKey: .nextnPredictLayers)
    }

    /// Checks the model type and, when present, the architecture name.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIPanguUltraMoeMtpError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIPanguUltraMoeMtpError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
    }

    /// Decodes the base `pangu_ultra_moe` configuration from the same JSON and checks everything
    /// the forward pass reads.
    ///
    /// The base struct checks its own `model_type` and `architectures`, which name the main model.
    /// Only those two keys are rewritten to the base values, for this decode only, so the MTP block
    /// is built with the main model's dimensions. The caller's JSON is not changed.
    public static func baseConfiguration(from data: Data) throws -> SharpAIPanguUltraMoeConfiguration {
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SharpAIPanguUltraMoeMtpError.invalidConfiguration("config must be a JSON object")
        }
        object["model_type"] = SharpAIPanguUltraMoeConfiguration.expectedModelType
        object["architectures"] = [SharpAIPanguUltraMoeConfiguration.expectedArchitecture]
        let baseData = try JSONSerialization.data(withJSONObject: object)
        let base = try JSONDecoder.json5().decode(
            SharpAIPanguUltraMoeConfiguration.self, from: baseData)
        try base.validateForForward()
        return base
    }
}
