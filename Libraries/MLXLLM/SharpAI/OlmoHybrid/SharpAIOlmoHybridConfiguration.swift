// SharpAI fork overlay: olmo_hybrid configuration. Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow transformers' configuration_olmo_hybrid.py (OlmoHybridConfig).
// The GatedDeltaNet init ranges (linear_a_log_*, linear_dt_*) only matter for training and are not read.

import Foundation
import MLXLMCommon

public struct SharpAIOlmoHybridConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let hiddenAct: String
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool
    public let layerTypes: [String]

    /// RoPE base from `rope_parameters.rope_theta`. Nil means NoPE: full attention runs without rotary embeddings.
    public let ropeTheta: Float?
    let ropeType: String

    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int
    public let linearAllowNegEigval: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case linearAllowNegEigval = "linear_allow_neg_eigval"
    }

    private struct RopeParameters: Decodable {
        let ropeType: String?
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "olmo_hybrid"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3840
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 30
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100352
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            // Default from OlmoHybridConfig: linear attention, full attention every 4th layer.
            let count = max(0, hiddenLayers)
            var types = (0 ..< count).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
            if !types.contains("full_attention"), !types.isEmpty {
                types[types.count - 1] = "full_attention"
            }
            layerTypes = types
        }

        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta
        ropeType = rope?.ropeType ?? "default"

        linearNumKeyHeads =
            try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? attentionHeads
        linearNumValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? attentionHeads
        linearKeyHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim)
            ?? (linearNumKeyHeads > 0 ? Int(0.75 * Double(hiddenSize) / Double(linearNumKeyHeads)) : 0)
        linearValueHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? (2 * linearKeyHeadDim)
        linearConvKernelDim =
            try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        linearAllowNegEigval =
            try c.decodeIfPresent(Bool.self, forKey: .linearAllowNegEigval) ?? true
    }

    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> ModelFactoryError {
            .invalidConfiguration("olmo_hybrid: \(message)")
        }

        let positive = [
            hiddenSize, hiddenLayers, intermediateSize, attentionHeads, kvHeads, headDim,
            vocabularySize, maxPositionEmbeddings, linearNumKeyHeads, linearNumValueHeads,
            linearKeyHeadDim, linearValueHeadDim, linearConvKernelDim,
        ]
        guard positive.allSatisfy({ $0 > 0 }) else {
            throw fail("hidden, layer, head, vocab and linear attention sizes must be positive")
        }
        guard linearConvKernelDim >= 2 else {
            throw fail("linear_conv_kernel_dim must be at least 2")
        }
        guard layerTypes.count == hiddenLayers else {
            throw fail("layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
        guard layerTypes.allSatisfy({ $0 == "linear_attention" || $0 == "full_attention" }) else {
            throw fail("layer_types must be linear_attention or full_attention")
        }
        guard layerTypes.contains("linear_attention") else {
            throw fail("at least one linear_attention layer is required")
        }
        guard layerTypes.contains("full_attention") else {
            throw fail("at least one full_attention layer is required")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw fail("num_attention_heads must be divisible by num_key_value_heads")
        }
        guard linearNumValueHeads % linearNumKeyHeads == 0 else {
            throw fail("linear_num_value_heads must be a multiple of linear_num_key_heads")
        }
        guard hiddenAct == "silu" else {
            throw fail("only silu hidden_act is supported, got \(hiddenAct)")
        }
        guard ropeType == "default" else {
            throw fail("only the default rope_type is supported, got \(ropeType)")
        }
    }
}
