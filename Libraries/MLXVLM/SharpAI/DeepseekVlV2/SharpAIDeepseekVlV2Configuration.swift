// SharpAI fork overlay: deepseek_vl_v2 configuration. Not part of upstream mlx-swift-lm.
//
// The text stack is a nested `language_config` block, decoded with DeepseekV2Configuration.
// Vision and projector fields are not decoded: the SigLIP vision tower and MlpProjector are not written yet.
// UNVERIFIED: the flat fallback and the MLA check follow the DeepSeek-VL2 config layout.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIDeepseekVlV2Error: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(SigLIP vision tower and non-MLA text attention missing)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after deepseek_vl_v2 sanitize: \(key)"
        }
    }
}

/// Text dimensions read from the language block. DeepseekV2Configuration keeps its fields internal.
public struct SharpAIDeepseekVlV2TextDimensions: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let kvLoraRank: Int?

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case kvLoraRank = "kv_lora_rank"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 102400
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 30
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 32
        kvLoraRank = try container.decodeIfPresent(Int.self, forKey: .kvLoraRank)
    }
}

public struct SharpAIDeepseekVlV2Configuration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textConfiguration: DeepseekV2Configuration
    public let textDimensions: SharpAIDeepseekVlV2TextDimensions
    public let hasVisionConfig: Bool

    /// True when the text stack sets kv_lora_rank (MLA). Standard attention is not implemented.
    public var textUsesMLA: Bool {
        (textDimensions.kvLoraRank ?? 0) > 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case languageConfig = "language_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "deepseek_vl_v2"

        if container.contains(.languageConfig) {
            textConfiguration = try container.decode(
                DeepseekV2Configuration.self, forKey: .languageConfig)
            textDimensions = try container.decode(
                SharpAIDeepseekVlV2TextDimensions.self, forKey: .languageConfig)
        } else {
            textConfiguration = try DeepseekV2Configuration(from: decoder)
            textDimensions = try SharpAIDeepseekVlV2TextDimensions(from: decoder)
        }

        hasVisionConfig = container.contains(.visionConfig)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.hiddenSize, text.numHiddenLayers, text.numAttentionHeads,
            text.numKeyValueHeads, text.vocabSize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_vl_v2: text hidden, layer, head and vocab sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "deepseek_vl_v2: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
