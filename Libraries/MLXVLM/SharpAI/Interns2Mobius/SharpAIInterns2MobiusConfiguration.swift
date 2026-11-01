// SharpAI fork overlay: interns2_mobius configuration. Not part of upstream mlx-swift-lm.
//
// Source: HF internlm/Intern-S2-Mobius config.json and configuration_interns2_mobius.py.
// Verified against: https://huggingface.co/internlm/Intern-S2-Mobius/resolve/main/config.json
// (model_type "interns2_mobius", architecture InternS2MobiusForConditionalGeneration).
// The text model is decoded from the nested "text_config" (or from the top level when that key is
// absent). The real config has no top-level "mtp_num_experts"; the value lives in text_config and is
// read there. vision_config is not mapped, because the vision tower is not ported.

import Foundation
import MLXLLM
import MLXLMCommon

/// Kept for callers that matched on it. The creator no longer throws it, because the text model exists.
public enum SharpAIInterns2MobiusError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) has no implemented model class "
                + "(architecture InternS2MobiusForConditionalGeneration, text stack only)."
        }
    }
}

public struct SharpAIInterns2MobiusConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "interns2_mobius"

    public let modelType: String
    public let architectures: [String]
    /// Top-level `mtp_num_experts`, if present. The real config keeps it under text_config (256).
    public let mtpNumExperts: Int
    /// Text model config, from "text_config" or from the top-level object when that key is absent.
    public let textConfiguration: SharpAIInterns2MobiusTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case mtpNumExperts = "mtp_num_experts"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        mtpNumExperts = try c.decodeIfPresent(Int.self, forKey: .mtpNumExperts) ?? 0
        if let nested = try c.decodeIfPresent(
            SharpAIInterns2MobiusTextConfiguration.self, forKey: .textConfig)
        {
            textConfiguration = nested
        } else {
            textConfiguration = try SharpAIInterns2MobiusTextConfiguration(from: decoder)
        }
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "interns2_mobius: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        guard mtpNumExperts >= 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "interns2_mobius: mtp_num_experts must not be negative, got \(mtpNumExperts)")
        }
        let text = textConfiguration
        guard text.hiddenLayers > 0, text.layerTypes.count == text.hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "interns2_mobius: layer_types must list one entry per hidden layer")
        }
        guard text.numBlocks > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "interns2_mobius: num_blocks must be positive, got \(text.numBlocks)")
        }
        guard text.numExpertsPerTok > 0, text.numExpertsPerTok <= text.numExperts else {
            throw ModelFactoryError.invalidConfiguration(
                "interns2_mobius: num_experts_per_tok must be in 1...num_experts")
        }
    }
}
