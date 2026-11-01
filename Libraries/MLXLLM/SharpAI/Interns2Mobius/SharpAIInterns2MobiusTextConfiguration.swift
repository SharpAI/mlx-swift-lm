// SharpAI fork overlay: interns2_mobius text configuration. Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow configuration_interns2_mobius.py and the real text_config of
// internlm/Intern-S2-Mobius (Verified against: https://huggingface.co/internlm/Intern-S2-Mobius/resolve/main/config.json).
// The config may nest these keys under "text_config" or sit at the top level; both decode the same way.

import Foundation
import MLXLMCommon

public struct SharpAIInterns2MobiusTextConfiguration: Decodable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let attentionBias: Bool
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool
    public let layerTypes: [String]

    /// RoPE base from `rope_parameters.rope_theta`.
    public let ropeTheta: Float
    /// Fraction of `head_dim` that gets rotary embeddings (`rope_parameters.partial_rotary_factor`).
    public let partialRotaryFactor: Float

    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int

    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    /// Number of shared routed-expert banks. Layer `i` uses bank `i % numBlocks`.
    public let numBlocks: Int

    /// Number of routed experts in the MTP head (`mtp_num_experts`). Read only, the MTP head is not ported.
    public let mtpNumExperts: Int

    /// Rotary dimensions per head: `int(head_dim * partial_rotary_factor)`.
    public var rotaryDim: Int { Int(Float(headDim) * partialRotaryFactor) }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case attentionBias = "attention_bias"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case partialRotaryFactor = "partial_rotary_factor"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numBlocks = "num_blocks"
        case mtpNumExperts = "mtp_num_experts"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        let partialRotaryFactor: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "interns2_mobius_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 251392
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        if let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = types
        } else {
            // Default from the HF config: full attention every 4th layer (index 3, 7, ...).
            layerTypes = (0 ..< max(0, hiddenLayers)).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" }
        }

        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 10_000_000
        partialRotaryFactor =
            try rope?.partialRotaryFactor
            ?? c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? 0.25

        linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
        linearNumValueHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 32
        linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
        linearValueHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4

        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 2560
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 8
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 512
        numBlocks = try c.decodeIfPresent(Int.self, forKey: .numBlocks) ?? 4
        mtpNumExperts = try c.decodeIfPresent(Int.self, forKey: .mtpNumExperts) ?? 256
    }
}
