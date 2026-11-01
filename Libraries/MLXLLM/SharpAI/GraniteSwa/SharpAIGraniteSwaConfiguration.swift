// SharpAI fork overlay: granite_swa configuration. Not part of upstream mlx-swift-lm.
//
// Source: transformers models/granite_swa/configuration_granite_swa.py (GraniteSWAConfig).
// Defaults follow that file. Attention sinks are always on for this model type.

import Foundation
import MLXLMCommon

public struct SharpAIGraniteSwaConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let tieWordEmbeddings: Bool
    public let slidingWindow: Int
    /// Per-layer `"full_attention"` or `"sliding_attention"`.
    public let layerTypes: [String]
    /// Per-layer RoPE base. A value of 0 marks a NoPE layer.
    public let layerRopeTheta: [Float]
    public let embeddingMultiplier: Float
    public let attentionMultiplier: Float
    public let residualMultiplier: Float
    public let logitsScaling: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case embeddingMultiplier = "embedding_multiplier"
        case attentionMultiplier = "attention_multiplier"
        case residualMultiplier = "residual_multiplier"
        case logitsScaling = "logits_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite_swa"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map { $0 % 4 == 0 ? "full_attention" : "sliding_attention" }
        layerRopeTheta =
            try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta)
            ?? Array(repeating: ropeTheta, count: hiddenLayers)
        embeddingMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1.0
        attentionMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .attentionMultiplier) ?? 1.0
        residualMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .residualMultiplier) ?? 1.0
        logitsScaling = try c.decodeIfPresent(Float.self, forKey: .logitsScaling) ?? 1.0
    }

    public func validateModelConfiguration() throws {
        let dims = [
            hiddenSize, hiddenLayers, intermediateSize, attentionHeads, kvHeads, vocabularySize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "granite_swa: hidden, layer, head and vocab sizes must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "granite_swa: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "granite_swa: layer_types must hold one full or sliding entry per layer")
        }
        guard layerRopeTheta.count == hiddenLayers, layerRopeTheta.allSatisfy({ $0 >= 0 })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "granite_swa: layer_rope_theta must hold one non-negative value per layer")
        }
        if layerTypes.contains("sliding_attention"), slidingWindow <= 0 {
            throw ModelFactoryError.invalidConfiguration(
                "granite_swa: sliding_window must be positive when a layer slides")
        }
    }
}
