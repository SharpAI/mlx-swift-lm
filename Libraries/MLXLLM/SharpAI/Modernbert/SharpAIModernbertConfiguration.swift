// SharpAI fork overlay: modernbert configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow the HF ModernBERT config. Only the fields the encoder reads are decoded.
// The fixture's position_embedding_type and classifier_* fields are ignored.

import Foundation
import MLXLMCommon

enum SharpAIModernbertError: LocalizedError {
    case invalidConfiguration(String)
    case generationUnsupported

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid modernbert configuration: \(message)"
        case .generationUnsupported:
            return
                "modernbert is an encoder. It has no autoregressive generation; use hiddenStates(_:attentionMask:)."
        }
    }
}

struct SharpAIModernbertConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var vocabularySize: Int
    var normEps: Float = 1e-5
    var normBias: Bool = false
    var attentionBias: Bool = false
    var mlpBias: Bool = false
    var hiddenActivation: String = "gelu"
    var localAttention: Int = 128
    var globalAttnEveryNLayers: Int = 3
    var globalRopeTheta: Float = 160_000
    var localRopeTheta: Float = 10_000
    var maxPositionEmbeddings: Int?
    var modelType: String?
    var architectures: [String] = []

    var headDimensions: Int {
        hiddenSize / attentionHeads
    }

    /// Half the local attention window. Each token sees this many positions on each side.
    var localHalfWindow: Int {
        localAttention / 2
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case normEps = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case hiddenActivation = "hidden_activation"
        case localAttention = "local_attention"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case modelType = "model_type"
        case architectures
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)

        if let normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) {
            self.normEps = normEps
        }
        if let normBias = try container.decodeIfPresent(Bool.self, forKey: .normBias) {
            self.normBias = normBias
        }
        if let attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) {
            self.attentionBias = attentionBias
        }
        if let mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) {
            self.mlpBias = mlpBias
        }
        if let hiddenActivation = try container.decodeIfPresent(
            String.self, forKey: .hiddenActivation)
        {
            self.hiddenActivation = hiddenActivation
        }
        if let localAttention = try container.decodeIfPresent(Int.self, forKey: .localAttention) {
            self.localAttention = localAttention
        }
        if let every = try container.decodeIfPresent(
            Int.self, forKey: .globalAttnEveryNLayers)
        {
            self.globalAttnEveryNLayers = every
        }
        if let globalRopeTheta = try container.decodeIfPresent(
            Float.self, forKey: .globalRopeTheta)
        {
            self.globalRopeTheta = globalRopeTheta
        }
        if let localRopeTheta = try container.decodeIfPresent(
            Float.self, forKey: .localRopeTheta)
        {
            self.localRopeTheta = localRopeTheta
        }
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
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
            ("vocab_size", vocabularySize),
            ("local_attention", localAttention),
            ("global_attn_every_n_layers", globalAttnEveryNLayers),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIModernbertError.invalidConfiguration("\(name) must be positive")
        }
        if hiddenSize % attentionHeads != 0 {
            throw SharpAIModernbertError.invalidConfiguration(
                "hidden_size must be divisible by num_attention_heads")
        }
        if headDimensions % 2 != 0 {
            throw SharpAIModernbertError.invalidConfiguration(
                "head dimension must be even for rotary embeddings")
        }
        if hiddenActivation != "gelu" {
            throw SharpAIModernbertError.invalidConfiguration(
                "hidden_activation \(hiddenActivation) is not supported; only gelu")
        }
    }
}
