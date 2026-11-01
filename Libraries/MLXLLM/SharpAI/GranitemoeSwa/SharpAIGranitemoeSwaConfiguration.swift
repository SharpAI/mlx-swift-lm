// SharpAI fork overlay: granitemoe_swa configuration. Not part of upstream mlx-swift-lm.
//
// Source: vLLM granitemoeshared.py (registry maps GraniteMoeSWAForCausalLM to it) and
// granite.py `granite_layer_attn_params`. Key names follow vLLM's attribute reads.
// UNVERIFIED: defaults for keys vLLM reads without a fallback (rms_norm_eps, rope_theta,
// the four multipliers, tie_word_embeddings) follow HF Granite defaults. Check them
// against a real granitemoe_swa config.json.

import Foundation
import MLXLMCommon

public struct SharpAIGranitemoeSwaConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    /// Per-expert hidden size (also the routed MoE width).
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let numLocalExperts: Int
    public let numExpertsPerToken: Int
    /// Width of the dense shared MLP. Nil or 0 means no shared MLP.
    public let sharedIntermediateSize: Int?
    public let maxPositionEmbeddings: Int?
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    /// Per-layer RoPE base. A value of 0 marks a NoPE layer.
    public let layerRopeTheta: [Float]?
    /// Per-layer `"full_attention"` or `"sliding_attention"`. Nil means full attention everywhere.
    public let layerTypes: [String]?
    public let slidingWindow: Int?
    /// Explicit sink switch. Nil means sinks are on whenever `layer_types` is set.
    public let attentionSinks: Bool?
    public let embeddingMultiplier: Float
    /// Nil falls back to `1 / head_dim`, as vLLM does.
    public let attentionMultiplier: Float?
    public let residualMultiplier: Float
    public let logitsScaling: Float
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case sharedIntermediateSize = "shared_intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case layerRopeTheta = "layer_rope_theta"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case attentionSinks = "attention_sinks"
        case embeddingMultiplier = "embedding_multiplier"
        case attentionMultiplier = "attention_multiplier"
        case residualMultiplier = "residual_multiplier"
        case logitsScaling = "logits_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granitemoe_swa"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        numLocalExperts = try c.decode(Int.self, forKey: .numLocalExperts)
        numExpertsPerToken = try c.decode(Int.self, forKey: .numExpertsPerToken)
        sharedIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize)
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        layerRopeTheta = try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        attentionSinks = try c.decodeIfPresent(Bool.self, forKey: .attentionSinks)
        embeddingMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1.0
        attentionMultiplier = try c.decodeIfPresent(Float.self, forKey: .attentionMultiplier)
        residualMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .residualMultiplier) ?? 1.0
        logitsScaling = try c.decodeIfPresent(Float.self, forKey: .logitsScaling) ?? 1.0
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }

    /// Per-layer type, with full attention for every layer when `layer_types` is absent.
    public var resolvedLayerTypes: [String] {
        layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
    }

    /// Sinks are on when `attention_sinks` is set, or when `layer_types` is set and it is not.
    public var usesAttentionSinks: Bool {
        attentionSinks ?? (layerTypes != nil)
    }

    public func validateModelConfiguration() throws {
        let dims = [
            vocabularySize, hiddenSize, intermediateSize, hiddenLayers,
            attentionHeads, kvHeads, numLocalExperts, numExpertsPerToken,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "granitemoe_swa: sizes, head counts and expert counts must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "granitemoe_swa: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard hiddenSize % attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "granitemoe_swa: hidden_size must be divisible by num_attention_heads")
        }
        guard numExpertsPerToken <= numLocalExperts else {
            throw ModelFactoryError.invalidConfiguration(
                "granitemoe_swa: num_experts_per_tok must not exceed num_local_experts")
        }
        if let sharedIntermediateSize, sharedIntermediateSize < 0 {
            throw ModelFactoryError.invalidConfiguration(
                "granitemoe_swa: shared_intermediate_size must not be negative")
        }
        if let layerTypes {
            guard layerTypes.count == hiddenLayers,
                layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" })
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "granitemoe_swa: layer_types must hold one full or sliding entry per layer")
            }
        }
        if resolvedLayerTypes.contains("sliding_attention") {
            guard let slidingWindow, slidingWindow > 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "granitemoe_swa: sliding_window must be positive when a layer slides")
            }
        }
        if let layerRopeTheta {
            guard layerRopeTheta.count == hiddenLayers, layerRopeTheta.allSatisfy({ $0 >= 0 })
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "granitemoe_swa: layer_rope_theta must hold one non-negative value per layer")
            }
        }
    }
}
