// SharpAI fork overlay: ovis2_6_moe configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the MoE language block and the vision block as dimensions only. No model class exists yet.
// Checked against the real AIDC-AI/Ovis2.6-30B-A3B config.json (model_type ovis2_6_moe): the top level
// has llm_config (model_type qwen3_moe, num_experts 128, num_experts_per_tok 8, 32 heads, 4 KV heads),
// vit_config (model_type siglip2_navit) and visual_vocab_size 65536. The top level also repeats the
// MoE fields, which this decoder does not read because llm_config takes priority.
// Still UNVERIFIED: the text_config alias and the flat-config fallback, which the real file does not use.

import Foundation
import MLXLMCommon

public enum SharpAIOvis26MoeError: LocalizedError {
    case forwardNotImplemented(String)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(MoE text decoder, visual tokenizer and visual embedding table missing)."
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after ovis2_6_moe sanitize: \(key)"
        }
    }
}

/// Text dimensions read from the language block.
public struct SharpAIOvis26MoeTextDimensions: Decodable, Sendable {
    public let modelType: String?
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        // UNVERIFIED: a missing KV head count is read as plain multi-head attention.
        numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        // Missing expert counts decode as 0 and fail validation.
        numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
    }
}

public struct SharpAIOvis26MoeConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textDimensions: SharpAIOvis26MoeTextDimensions
    public let hasVisionConfig: Bool
    public let visualVocabSize: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case llmConfig = "llm_config"
        case textConfig = "text_config"
        case vitConfig = "vit_config"
        case visionConfig = "vision_config"
        case visualVocabSize = "visual_vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "ovis2_6_moe"

        if container.contains(.llmConfig) {
            textDimensions = try container.decode(
                SharpAIOvis26MoeTextDimensions.self, forKey: .llmConfig)
        } else if container.contains(.textConfig) {
            textDimensions = try container.decode(
                SharpAIOvis26MoeTextDimensions.self, forKey: .textConfig)
        } else {
            textDimensions = try SharpAIOvis26MoeTextDimensions(from: decoder)
        }

        hasVisionConfig =
            container.contains(.vitConfig) || container.contains(.visionConfig)
        visualVocabSize = try container.decodeIfPresent(Int.self, forKey: .visualVocabSize)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.vocabSize, text.hiddenSize, text.numHiddenLayers,
            text.numAttentionHeads, text.numKeyValueHeads,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "ovis2_6_moe: text vocab, hidden, layer and head sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "ovis2_6_moe: num_attention_heads must be divisible by num_key_value_heads")
        }
        // The dense variant is gated and out of scope: only the MoE text block is accepted.
        guard text.numExperts > 0,
            text.numExpertsPerTok > 0,
            text.numExpertsPerTok <= text.numExperts
        else {
            throw ModelFactoryError.invalidConfiguration(
                "ovis2_6_moe: requires num_experts > 0 and 0 < num_experts_per_tok <= num_experts")
        }
        guard hasVisionConfig else {
            throw ModelFactoryError.invalidConfiguration(
                "ovis2_6_moe: vit_config (or vision_config) is required")
        }
        if let visualVocabSize, visualVocabSize <= 0 {
            throw ModelFactoryError.invalidConfiguration(
                "ovis2_6_moe: visual_vocab_size must be positive when set")
        }
    }
}
