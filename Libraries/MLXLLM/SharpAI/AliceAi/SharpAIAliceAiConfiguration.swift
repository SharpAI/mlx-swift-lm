// SharpAI fork overlay: alice_ai configuration. Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow vLLM's alice_ai.py and the checkpoint config in
// scripts/conformance/offline/fixtures/alice_ai. Keys not listed here (mtp_*,
// number_of_conv_states, kda_allow_negative_eigenvalues, ...) are not read.

import Foundation
import MLXLMCommon

public struct SharpAIAliceAiConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let layerTypes: [String]
    public let hiddenAct: String
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool

    public let linearNumKeyHeads: Int
    public let linearNumValueHeads: Int
    public let linearKeyHeadDim: Int
    public let linearValueHeadDim: Int
    public let linearConvKernelDim: Int

    /// AttnRes grouping: a completed block closes every this many layers.
    public let attnResBlockSize: Int

    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let normTopkProb: Bool
    public let routerScoreFunction: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case layerTypes = "layer_types"
        case hiddenAct = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case attnResBlockSize = "block_attn_res_block_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case routerScoreFunction = "router_score_function"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "alice_ai"
        hiddenSize = hidden
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = heads
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / heads)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        partialRotaryFactor = try c.decode(Float.self, forKey: .partialRotaryFactor)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try c.decode(Int.self, forKey: .linearConvKernelDim)

        attnResBlockSize = try c.decode(Int.self, forKey: .attnResBlockSize)

        numExperts = try c.decode(Int.self, forKey: .numExperts)
        numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(Int.self, forKey: .sharedExpertIntermediateSize)
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routerScoreFunction =
            try c.decodeIfPresent(String.self, forKey: .routerScoreFunction) ?? "sigmoid"
    }

    public func validateModelConfiguration() throws {
        func fail(_ message: String) -> ModelFactoryError {
            .invalidConfiguration("alice_ai: \(message)")
        }

        let positive = [
            hiddenSize, hiddenLayers, attentionHeads, kvHeads, headDim, vocabularySize,
            maxPositionEmbeddings, linearNumKeyHeads, linearKeyHeadDim, linearConvKernelDim,
            attnResBlockSize, numExperts, numExpertsPerTok, moeIntermediateSize,
            sharedExpertIntermediateSize,
        ]
        guard positive.allSatisfy({ $0 > 0 }) else {
            throw fail("hidden, layer, head, vocab, expert and block sizes must be positive")
        }
        guard layerTypes.count == hiddenLayers else {
            throw fail("layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
        guard layerTypes.allSatisfy({ $0 == "linear_attention" || $0 == "full_attention" }) else {
            throw fail("layer_types must be linear_attention or full_attention")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw fail("num_attention_heads must be divisible by num_key_value_heads")
        }
        guard linearNumKeyHeads == linearNumValueHeads, linearKeyHeadDim == linearValueHeadDim,
            linearValueHeadDim > 0, linearNumValueHeads > 0
        else {
            throw fail("KDA needs equal key and value head counts and dimensions")
        }
        guard numExpertsPerTok <= numExperts else {
            throw fail("num_experts_per_tok must not exceed num_experts")
        }
        guard hiddenAct == "silu" else {
            throw fail("only silu hidden_act is supported, got \(hiddenAct)")
        }
        guard routerScoreFunction == "sigmoid" else {
            throw fail("only sigmoid router_score_function is supported, got \(routerScoreFunction)")
        }
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1,
            Int(Float(headDim) * partialRotaryFactor) > 0
        else {
            throw fail("partial_rotary_factor must give at least one rotary dimension")
        }
    }
}
