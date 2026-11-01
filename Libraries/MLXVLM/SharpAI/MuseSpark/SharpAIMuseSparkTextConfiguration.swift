// SharpAI fork overlay: muse_spark text backbone configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the `text_config` block of a muse_spark checkpoint. Field names follow
// transformers modeling_muse_spark.py (MuseSparkTextModel, MuseSparkTextAttention, MuseSparkMoE).
// Vision and audio configs are not decoded here. `num_shared_experts` is not decoded because
// the reference MoE block does not use it.
// Lives in MLXVLM because SharpAIMuseSparkError is defined in this target.

import Foundation
import MLXLMCommon

struct SharpAIMuseSparkTextConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numHiddenLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let rmsNormEps: Float
    let postNormEps: Float
    let attentionBias: Bool
    let slidingWindow: Int
    let layerTypes: [String]
    /// Per-layer RoPE switch: a zero entry marks a NoPE layer. The non-zero value itself is not
    /// used, matching the reference, which applies `rope_parameters.rope_theta` to every RoPE layer.
    let layerRopeTheta: [Float]
    let mlpLayerTypes: [String]
    let moeHiddenSize: Int
    let moeIntermediateSize: Int
    let numLocalExperts: Int
    let numExpertsPerTok: Int
    let routedScalingFactor: Float
    let outputMultiplier: Float
    let finalLogitSoftcapping: Float
    let residualGateTemperature: Float
    let qkScaleFactor: Float
    let ropeTheta: Float
    let initializerRange: Float
    let hiddenAct: String
    let tieWordEmbeddings: Bool

    private struct RopeParameters: Decodable {
        let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case mlpLayerTypes = "mlp_layer_types"
        case moeHiddenSize = "moe_hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case residualGateTemperature = "residual_gate_temperature"
        case qkScaleFactor = "qk_scale_factor"
        case ropeParameters = "rope_parameters"
        case initializerRange = "initializer_range"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .numAttentionHeads)

        hiddenSize = hidden
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = heads
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (heads > 0 ? hidden / heads : 0)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        postNormEps = try container.decode(Float.self, forKey: .postNormEps)
        attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        slidingWindow = try container.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        layerRopeTheta = try container.decode([Float].self, forKey: .layerRopeTheta)
        mlpLayerTypes = try container.decode([String].self, forKey: .mlpLayerTypes)
        moeHiddenSize = try container.decode(Int.self, forKey: .moeHiddenSize)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        numLocalExperts = try container.decode(Int.self, forKey: .numLocalExperts)
        numExpertsPerTok = try container.decode(Int.self, forKey: .numExpertsPerTok)
        routedScalingFactor = try container.decode(Float.self, forKey: .routedScalingFactor)
        outputMultiplier = try container.decode(Float.self, forKey: .outputMultiplier)
        finalLogitSoftcapping = try container.decode(Float.self, forKey: .finalLogitSoftcapping)
        residualGateTemperature = try container.decode(Float.self, forKey: .residualGateTemperature)
        qkScaleFactor = try container.decode(Float.self, forKey: .qkScaleFactor)
        ropeTheta = try container.decode(RopeParameters.self, forKey: .ropeParameters).ropeTheta
        initializerRange =
            try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        try validate()
    }

    private func validate() throws {
        guard hiddenAct == "silu" else {
            throw SharpAIMuseSparkError.unsupported("text hidden_act \(hiddenAct) is not implemented")
        }
        guard !tieWordEmbeddings else {
            throw SharpAIMuseSparkError.unsupported("tie_word_embeddings is not implemented")
        }
        guard numHiddenLayers > 0, numAttentionHeads > 0, numKeyValueHeads > 0, headDim > 0,
            numAttentionHeads % numKeyValueHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_spark: heads must divide evenly into key/value heads and layers must be > 0")
        }
        guard layerTypes.count == numHiddenLayers,
            layerRopeTheta.count == numHiddenLayers,
            mlpLayerTypes.count == numHiddenLayers
        else {
            throw ModelFactoryError.invalidConfiguration(
                "muse_spark: layer_types, layer_rope_theta and mlp_layer_types must have num_hidden_layers entries")
        }
        guard layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" }) else {
            throw SharpAIMuseSparkError.unsupported("unknown attention layer type in layer_types")
        }
        guard mlpLayerTypes.allSatisfy({ $0 == "sparse" || $0 == "dense" }) else {
            throw SharpAIMuseSparkError.unsupported("unknown mlp layer type in mlp_layer_types")
        }
        if mlpLayerTypes.contains("sparse") {
            guard numExpertsPerTok > 0, numExpertsPerTok <= numLocalExperts else {
                throw ModelFactoryError.invalidConfiguration(
                    "muse_spark: num_experts_per_tok must be in 1...num_local_experts")
            }
        }
    }
}
