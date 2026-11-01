// SharpAI fork overlay: openai_privacy_filter configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow transformers OpenAIPrivacyFilterConfig
// (configuration_openai_privacy_filter.py). The label count comes from id2label.

import Foundation
import MLXLMCommon

enum SharpAIOpenaiPrivacyFilterError: LocalizedError {
    case invalidConfiguration(String)
    case tokenClassifierOnly

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            return "Invalid openai_privacy_filter configuration: \(message)"
        case .tokenClassifierOnly:
            return
                "openai_privacy_filter is a token classifier. It has no autoregressive generation; use tokenLogits(_:attentionMask:)."
        }
    }
}

struct SharpAIOpenaiPrivacyFilterConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// The `rope_parameters` block. Only the YaRN fields this model reads.
    struct RopeParameters: Decodable, Sendable {
        var ropeType: String = "yarn"
        var ropeTheta: Float = 150_000
        var factor: Float = 32
        var betaFast: Float = 32
        var betaSlow: Float = 1
        var originalMaxPositionEmbeddings: Int = 4096
        var truncate: Bool = false

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
            case factor
            case betaFast = "beta_fast"
            case betaSlow = "beta_slow"
            case originalMaxPositionEmbeddings = "original_max_position_embeddings"
            case truncate
        }

        init() {}

        init(from decoder: Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "yarn"
            ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 150_000
            factor = try container.decodeIfPresent(Float.self, forKey: .factor) ?? 32
            betaFast = try container.decodeIfPresent(Float.self, forKey: .betaFast) ?? 32
            betaSlow = try container.decodeIfPresent(Float.self, forKey: .betaSlow) ?? 1
            originalMaxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings)
                ?? 4096
            truncate = try container.decodeIfPresent(Bool.self, forKey: .truncate) ?? false
        }
    }

    var hiddenLayers: Int
    var localExperts: Int
    var expertsPerToken: Int
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var headDimensions: Int
    var attentionHeads: Int
    var kvHeads: Int
    var slidingWindow: Int
    var rmsNormEps: Float
    var attentionBias: Bool
    var rope: RopeParameters
    var numLabels: Int

    enum CodingKeys: String, CodingKey {
        case hiddenLayers = "num_hidden_layers"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case headDimensions = "head_dim"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case attentionBias = "attention_bias"
        case ropeParameters = "rope_parameters"
        case id2label
        case numLabels = "num_labels"
    }

    init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 8
        localExperts = try container.decodeIfPresent(Int.self, forKey: .localExperts) ?? 128
        expertsPerToken = try container.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 4
        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200_064
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 640
        intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 640
        // transformers defaults head_dim to 64 when the key is absent.
        headDimensions = try container.decodeIfPresent(Int.self, forKey: .headDimensions) ?? 64
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 14
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 128
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        rope =
            try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            ?? RopeParameters()

        let id2label = try container.decodeIfPresent([String: String].self, forKey: .id2label)
        numLabels =
            try container.decodeIfPresent(Int.self, forKey: .numLabels)
            ?? id2label?.count ?? 33
    }

    func validateModelConfiguration() throws {
        let positive: [(String, Int)] = [
            ("hidden_size", hiddenSize),
            ("num_hidden_layers", hiddenLayers),
            ("intermediate_size", intermediateSize),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads),
            ("head_dim", headDimensions),
            ("num_local_experts", localExperts),
            ("num_experts_per_tok", expertsPerToken),
            ("vocab_size", vocabularySize),
            ("num_labels", numLabels),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration("\(name) must be positive")
        }
        if attentionHeads % kvHeads != 0 {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration(
                "num_attention_heads must be divisible by num_key_value_heads")
        }
        if headDimensions % 2 != 0 {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration(
                "head_dim must be even for rotary embeddings")
        }
        if expertsPerToken > localExperts {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration(
                "num_experts_per_tok must not exceed num_local_experts")
        }
        if slidingWindow < 0 {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration(
                "sliding_window must not be negative")
        }
        if rope.ropeType != "yarn" {
            throw SharpAIOpenaiPrivacyFilterError.invalidConfiguration(
                "rope_type \(rope.ropeType) is not supported; only yarn")
        }
    }
}
