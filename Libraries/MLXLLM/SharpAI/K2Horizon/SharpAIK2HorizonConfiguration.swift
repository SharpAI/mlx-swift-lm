// SharpAI fork overlay: k2_horizon configuration. Not part of upstream mlx-swift-lm.
//
// Llama-shaped text config with two additions:
// - `rope_parameters` (rope_theta and scaling in one object). LlamaConfiguration ignores it.
// - `rms_norm_groups`, the group count for grouped RMSNorm. Default 1 is plain RMSNorm.
//
// UNVERIFIED: the key names `rope_parameters` (as used by transformers v5 style configs) and
// `rms_norm_groups` are assumed. No K2 Horizon config.json was available to check them.

import Foundation
import MLXLMCommon

public struct SharpAIK2HorizonConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let headDimensions: Int?
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let kvHeads: Int
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let rmsNormGroups: Int

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case rmsNormGroups = "rms_norm_groups"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "k2_horizon"
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        rmsNormGroups = try container.decodeIfPresent(Int.self, forKey: .rmsNormGroups) ?? 1

        // rope_parameters takes precedence over top-level rope_theta and rope_scaling.
        var theta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        var scaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if let parameters = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            if let parameterTheta = parameters["rope_theta"]?.asFloat() {
                theta = parameterTheta
            }
            var scalingParameters = parameters
            scalingParameters["rope_theta"] = nil

            var ropeType = "default"
            if case .string(let name)? = scalingParameters["rope_type"] ?? scalingParameters["type"]
            {
                ropeType = name
            }
            scaling = ropeType == "default" ? nil : scalingParameters
        }
        ropeTheta = theta
        ropeScaling = scaling

        guard rmsNormGroups > 0, hiddenSize % rmsNormGroups == 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .rmsNormGroups, in: container,
                debugDescription: "rms_norm_groups must be positive and divide hidden_size")
        }
    }

    /// The Llama view of this configuration. Used to build the shared attention and MLP blocks.
    var llama: LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: hiddenSize,
            hiddenLayers: hiddenLayers,
            intermediateSize: intermediateSize,
            attentionHeads: attentionHeads,
            headDimensions: headDimensions,
            rmsNormEps: rmsNormEps,
            vocabularySize: vocabularySize,
            kvHeads: kvHeads,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeTheta: ropeTheta,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionBias: attentionBias,
            mlpBias: mlpBias,
            modelType: modelType
        )
    }
}
