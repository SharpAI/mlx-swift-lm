// SharpAI fork overlay: hyperclovax configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow scripts/conformance/offline/fixtures/hyperclovax/
// naver-hyperclovax__HyperCLOVAX-SEED-Think-14B.json.
// UNVERIFIED: defaults for keys the fixture lacks, and logits_scaling as a divisor (as in Granite).

import Foundation
import MLXLMCommon

public enum SharpAIHyperclovaxError: Error, Equatable {
    case invalidConfiguration(String)
}

public struct SharpAIHyperclovaxConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let hiddenAct: String
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    /// Scale applied to query-key products. The fixture sets it to `1 / head_dim`.
    /// UNVERIFIED: the fallback `head_dim ** -0.5` is a guess for checkpoints that omit it.
    public let attentionMultiplier: Float
    public let embeddingMultiplier: Float
    public let residualMultiplier: Float
    public let logitsScaling: Float
    /// Adds an RMSNorm to each sublayer output before its residual add.
    /// UNVERIFIED: the placement and checkpoint key names are not in the repo.
    public let usePostNorm: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case logitsScaling = "logits_scaling"
        case usePostNorm = "use_post_norm"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)
        let resolvedHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (heads > 0 ? hidden / heads : 0)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "hyperclovax"
        hiddenSize = hidden
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = resolvedHeadDim
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        attentionMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .attentionMultiplier)
            ?? pow(Float(resolvedHeadDim), -0.5)
        embeddingMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .embeddingMultiplier) ?? 1.0
        residualMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .residualMultiplier) ?? 1.0
        logitsScaling = try container.decodeIfPresent(Float.self, forKey: .logitsScaling) ?? 1.0
        usePostNorm = try container.decodeIfPresent(Bool.self, forKey: .usePostNorm) ?? false
    }

    public func validateModelConfiguration() throws {
        guard hiddenSize > 0, hiddenLayers > 0, intermediateSize > 0, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, headDim > 0
        else {
            throw SharpAIHyperclovaxError.invalidConfiguration(
                "sizes, head counts and vocabulary must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIHyperclovaxError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIHyperclovaxError.invalidConfiguration(
                "unsupported hidden_act \(hiddenAct)")
        }
        guard logitsScaling != 0 else {
            throw SharpAIHyperclovaxError.invalidConfiguration("logits_scaling must be non-zero")
        }
    }
}
