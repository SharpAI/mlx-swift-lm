//
//  SharpAIStep1Configuration.swift
//  MLXVLM
//
//  Config for `step1` (Step1ForCausalLM).
//  Field names and defaults follow stepfun-ai/Step-Audio-EditX configuration_step1.py and
//  its config.json. The KV head count is `num_attention_groups` in that checkpoint.
//  There is no rotary config: the reference attention has no RoPE, and `rope_theta` is
//  decoded only for compatibility and is not used by the model.
//  The audio tokenizer/encoder config is not typed here.
//

import Foundation
import MLXLMCommon

enum SharpAIStep1Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "step1 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "step1 invalid configuration: \(message)"
        }
    }
}

struct SharpAIStep1Configuration: Codable, Sendable, ModelConfigurationValidating {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    /// KV head count, read from `num_attention_groups` (the checkpoint key), then
    /// `num_key_value_heads`. Defaults to `attentionHeads` when both are absent.
    /// The Python `Step1Config` default is 8; that is not used here.
    var kvHeads: Int
    /// Default 1e-5 when absent (matches Step1Config).
    var rmsNormEps: Float
    /// Unused by the model: the reference has no RoPE. Default 10000 when absent.
    var ropeTheta: Float
    var maxPositionEmbeddings: Int?
    /// Default false when absent (matches PretrainedConfig).
    var tieWordEmbeddings: Bool
    /// Activation name as written in config.json. The MLP always uses SiLU, as in the reference.
    var hiddenAct: String?

    var headDim: Int { hiddenSize / attentionHeads }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "step1"
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let groupContainer = try decoder.container(keyedBy: KVHeadGroupKeys.self)
        kvHeads =
            try groupContainer.decodeIfPresent(Int.self, forKey: .numAttentionGroups)
            ?? container.decodeIfPresent(Int.self, forKey: .kvHeads)
            ?? attentionHeads
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
    }

    /// Decode-only key for the checkpoint's KV head count. It is kept out of `CodingKeys`
    /// because it has no stored property, so `Encodable` synthesis would not match it.
    private enum KVHeadGroupKeys: String, CodingKey {
        case numAttentionGroups = "num_attention_groups"
    }

    public func validateModelConfiguration() throws {
        let positive: [(String, Int)] = [
            ("vocab_size", vocabularySize),
            ("hidden_size", hiddenSize),
            ("intermediate_size", intermediateSize),
            ("num_hidden_layers", hiddenLayers),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIStep1Error.invalidConfiguration("\(name) must be positive, got \(value)")
        }
        guard hiddenSize % attentionHeads == 0 else {
            throw SharpAIStep1Error.invalidConfiguration(
                "hidden_size \(hiddenSize) is not divisible by num_attention_heads \(attentionHeads)")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIStep1Error.invalidConfiguration(
                "num_attention_heads \(attentionHeads) is not divisible by num_key_value_heads \(kvHeads)")
        }
    }
}
