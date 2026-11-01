//
//  SharpAIJais2Configuration.swift
//  MLXLLM
//
//  Text config for `jais2` (Jais 2, Jais2ForCausalLM).
//  Defaults follow Hugging Face transformers `Jais2Config`
//  (src/transformers/models/jais2/configuration_jais2.py, main branch):
//  attention_bias=True, mlp_bias=True, layer_norm_eps=1e-5, hidden_act="relu2",
//  tie_word_embeddings=False. The real checkpoint config.json (gated) was not
//  fetched, so the per-checkpoint values remain unverified.
//

import Foundation
import MLXLMCommon

enum SharpAIJais2Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "jais2 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "jais2 invalid configuration: \(message)"
        }
    }
}

struct SharpAIJais2Configuration: Codable, Sendable, ModelConfigurationValidating {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    /// Defaults to `attentionHeads` (no grouped-query attention) when absent.
    var kvHeads: Int
    /// transformers default 1e-5 when absent.
    var layerNormEps: Float
    /// transformers default base 10000 when absent. Read from top-level `rope_theta`
    /// or from `rope_parameters.rope_theta`.
    var ropeTheta: Float
    var maxPositionEmbeddings: Int?
    /// transformers default false when absent.
    var tieWordEmbeddings: Bool
    /// transformers default true when absent. Applies to q/k/v/o projections.
    var attentionBias: Bool
    /// transformers default true when absent. Applies to up/down projections.
    var mlpBias: Bool
    /// Explicit `head_dim`. Nil means `hiddenSize / attentionHeads`.
    var explicitHeadDim: Int?
    /// transformers default "relu2" when absent. Only relu2 is ported.
    var hiddenAct: String?

    var headDim: Int { explicitHeadDim ?? (hiddenSize / attentionHeads) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case explicitHeadDim = "head_dim"
        case hiddenAct = "hidden_act"
    }

    /// Only the `rope_theta` key of a `rope_parameters` object is read.
    private struct RopeParameters: Decodable {
        var ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "jais2"
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        layerNormEps =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        if let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = topLevelTheta
        } else if let parameters = try container.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters),
            let nestedTheta = parameters.ropeTheta
        {
            ropeTheta = nestedTheta
        } else {
            ropeTheta = 10_000
        }
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? true
        explicitHeadDim = try container.decodeIfPresent(Int.self, forKey: .explicitHeadDim)
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct)
    }

    /// Explicit encoder. `CodingKeys` carries `rope_parameters` for decoding only, which
    /// has no stored property, so synthesis is unavailable. Encoding writes the resolved
    /// top-level `rope_theta`, which `init(from:)` reads back.
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(layerNormEps, forKey: .layerNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encodeIfPresent(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(mlpBias, forKey: .mlpBias)
        try container.encodeIfPresent(explicitHeadDim, forKey: .explicitHeadDim)
        try container.encodeIfPresent(hiddenAct, forKey: .hiddenAct)
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
            throw SharpAIJais2Error.invalidConfiguration("\(name) must be positive, got \(value)")
        }
        if let explicitHeadDim {
            guard explicitHeadDim > 0 else {
                throw SharpAIJais2Error.invalidConfiguration(
                    "head_dim must be positive, got \(explicitHeadDim)")
            }
        } else {
            guard hiddenSize % attentionHeads == 0 else {
                throw SharpAIJais2Error.invalidConfiguration(
                    "hidden_size \(hiddenSize) is not divisible by num_attention_heads \(attentionHeads)")
            }
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIJais2Error.invalidConfiguration(
                "num_attention_heads \(attentionHeads) is not divisible by num_key_value_heads \(kvHeads)")
        }
    }
}
