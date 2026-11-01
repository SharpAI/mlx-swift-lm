// SharpAI fork overlay: jina_embeddings_v5 nano configuration. Not part of upstream mlx-swift-lm.
// The nano variant is a bidirectional EuroBERT encoder with `is_decoder: false`
// (vLLM jina.py / modeling_eurobert.py). Its config has no `head_dim`, so it cannot go through
// Qwen3Configuration, which requires that key. Defaults follow EuroBertConfig:
// rope_theta 250000, rms_norm_eps 1e-5, max_position_embeddings 8192, no attention or MLP bias.

import Foundation
import MLXLMCommon

enum SharpAIJinaEmbeddingsV5NanoError: LocalizedError {
    case invalidConfiguration(String)
    case generationUnsupported

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid jina_embeddings_v5 nano configuration: \(message)"
        case .generationUnsupported:
            return
                "jina_embeddings_v5 nano is a bidirectional encoder. It has no autoregressive generation; use hiddenStates(_:)."
        }
    }
}

struct SharpAIJinaEmbeddingsV5NanoConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var maxPositionEmbeddings: Int
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool
    var mlpBias: Bool
    var tieWordEmbeddings: Bool
    var isDecoder: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case isDecoder = "is_decoder"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 250_000
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        isDecoder = try container.decodeIfPresent(Bool.self, forKey: .isDecoder) ?? false
    }

    func validateModelConfiguration() throws {
        let positive: [(String, Int)] = [
            ("hidden_size", hiddenSize),
            ("num_hidden_layers", hiddenLayers),
            ("intermediate_size", intermediateSize),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads),
            ("head_dim", headDim),
            ("vocab_size", vocabularySize),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIJinaEmbeddingsV5NanoError.invalidConfiguration("\(name) must be positive")
        }
        if attentionHeads % kvHeads != 0 {
            throw SharpAIJinaEmbeddingsV5NanoError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        if isDecoder {
            throw SharpAIJinaEmbeddingsV5NanoError.invalidConfiguration(
                "is_decoder must be false for the bidirectional nano encoder")
        }
        try validateRoPEConfiguration(
            ropeScaling, context: "SharpAIJinaEmbeddingsV5NanoConfiguration.rope_scaling")
    }
}
