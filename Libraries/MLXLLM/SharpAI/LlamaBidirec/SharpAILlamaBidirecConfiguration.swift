// SharpAI fork overlay: llama_bidirec configuration. Not part of upstream mlx-swift-lm.
//
// Same field names as LlamaConfiguration (HF Llama config keys). Kept as a separate type so
// the upstream Llama configuration is not changed.

import Foundation
import MLXLMCommon

enum SharpAILlamaBidirecError: LocalizedError {
    case invalidConfiguration(String)
    case generationUnsupported

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid llama_bidirec configuration: \(message)"
        case .generationUnsupported:
            return
                "llama_bidirec is a bidirectional encoder. It has no autoregressive generation; use hiddenStates(_:)."
        }
    }
}

struct SharpAILlamaBidirecConfiguration: Codable, Sendable, ModelConfigurationValidating {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDimensions: Int?
    var vocabularySize: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float = 10_000
    var ropeTraditional: Bool = false
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var tieWordEmbeddings: Bool = true
    var modelType: String?
    var architectures: [String] = []

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case modelType = "model_type"
        case architectures
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        if let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            self.ropeTheta = ropeTheta
        }
        if let ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
        {
            self.ropeTraditional = ropeTraditional
        }
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }
        if let tieWordEmbeddings = try container.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings)
        {
            self.tieWordEmbeddings = tieWordEmbeddings
        }
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }

    func validateModelConfiguration() throws {
        let positive: [(String, Int)] = [
            ("hidden_size", hiddenSize),
            ("num_hidden_layers", hiddenLayers),
            ("intermediate_size", intermediateSize),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads),
            ("vocab_size", vocabularySize),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAILlamaBidirecError.invalidConfiguration("\(name) must be positive")
        }
        if attentionHeads % kvHeads != 0 {
            throw SharpAILlamaBidirecError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        if headDimensions == nil && hiddenSize % attentionHeads != 0 {
            throw SharpAILlamaBidirecError.invalidConfiguration(
                "hidden_size must be divisible by num_attention_heads when head_dim is absent")
        }
    }
}
