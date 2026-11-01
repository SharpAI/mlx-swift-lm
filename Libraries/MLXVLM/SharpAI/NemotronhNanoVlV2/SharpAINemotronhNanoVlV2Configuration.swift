// SharpAI fork overlay: NemotronH_Nano_VL_V2 configuration. Not part of upstream mlx-swift-lm.
//
// The text stack is a nested `llm_config` block, decoded with the existing NemotronHConfiguration.
// Vision fields are not decoded: the RADIO vision tower has no reference source here.
// UNVERIFIED: the top-level flat fallback is a guess for configs that lack `llm_config`.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAINemotronhNanoVlV2Error: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)
    case mergedForwardNotImplemented

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its checkpoint loader is not wired yet "
                + "(vision keys are dropped by the sanitizer)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after NemotronH_Nano_VL_V2 sanitize: \(key)"
        case .mergedForwardNotImplemented:
            return
                "NemotronH_Nano_VL_V2 image+text forward needs an inputs_embeds entry point "
                + "on NemotronHModel, which is not exposed yet."
        }
    }
}

public struct SharpAINemotronhNanoVlV2Configuration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textConfiguration: NemotronHConfiguration
    public let hasVisionConfig: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case llmConfig = "llm_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "NemotronH_Nano_VL_V2"

        if container.contains(.llmConfig) {
            textConfiguration = try container.decode(
                NemotronHConfiguration.self, forKey: .llmConfig)
        } else {
            textConfiguration = try NemotronHConfiguration(from: decoder)
        }

        hasVisionConfig = container.contains(.visionConfig)
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let dims = [
            text.hiddenSize, text.numHiddenLayers, text.numAttentionHeads,
            text.numKeyValueHeads, text.vocabSize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "NemotronH_Nano_VL_V2: text hidden, layer, head and vocab sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "NemotronH_Nano_VL_V2: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
