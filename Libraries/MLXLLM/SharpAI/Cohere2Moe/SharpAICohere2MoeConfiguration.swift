// SharpAI fork overlay: cohere2_moe configuration. Not part of upstream mlx-swift-lm.
//
// Source: transformers v5.10.2, src/transformers/models/cohere2_moe/configuration_cohere2_moe.py
// and modeling_cohere2_moe.py. Defaults below follow that source, except where noted.
// UNVERIFIED against a real cohere2_moe checkpoint config.
// Deviations: num_experts has no default here (absent means no routed experts, all layers dense),
// where HF defaults to 8. Routed experts use intermediate_size, as in HF; moe_intermediate_size
// is decoded but unused by the model.

import Foundation
import MLXLMCommon

public enum SharpAICohere2MoeError: LocalizedError {
    case missingExpertWeight(String)
    case unexpectedExpertLayout(String)

    public var errorDescription: String? {
        switch self {
        case .missingExpertWeight(let key):
            return "Missing expert weight during sanitize: \(key)"
        case .unexpectedExpertLayout(let key):
            return "Unexpected expert weight layout during sanitize: \(key)"
        }
    }
}

public struct SharpAICohere2MoeConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    /// HF dataclass default is 128 (`head_dim: int = 128`).
    public let headDim: Int
    public let layerNormEps: Float
    /// When set, the norm is RMSNorm with this epsilon. When nil, it is the bias-free LayerNorm.
    public let rmsNormEps: Float?
    public let vocabularySize: Int
    public let logitScale: Float
    public let ropeTheta: Float
    public let slidingWindow: Int?
    public let slidingWindowPattern: Int
    public let prefixDenseSlidingWindowPattern: Int
    /// One of `sliding_attention` or `full_attention` per layer.
    public let layerTypes: [String]
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let firstKDenseReplace: Int
    public let prefixDenseIntermediateSize: Int?

    public let numExperts: Int?
    public let numExpertsPerTok: Int
    /// Decoded for compatibility. The model does not read it: HF routes experts at intermediate_size.
    public let moeIntermediateSize: Int?
    public let numSharedExperts: Int
    /// `average` (default) or `sum`.
    public let sharedExpertCombinationStrategy: String
    /// `softmax` (default) or `sigmoid`.
    public let expertSelectionFn: String
    public let normTopkProb: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case layerNormEps = "layer_norm_eps"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case logitScale = "logit_scale"
        case ropeTheta = "rope_theta"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case prefixDenseSlidingWindowPattern = "prefix_dense_sliding_window_pattern"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case firstKDenseReplace = "first_k_dense_replace"
        case prefixDenseIntermediateSize = "prefix_dense_intermediate_size"
        case numExperts = "num_experts"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case sharedExpertCombinationStrategy = "shared_expert_combination_strategy"
        case expertSelectionFn = "expert_selection_fn"
        case normTopkProb = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere2_moe"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionHeads = heads
        let kv = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        kvHeads = kv
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? 0.0625
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0

        // An absent key takes the HF default of 4096. An explicit null disables the window.
        if c.contains(.slidingWindow) {
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        } else {
            slidingWindow = 4096
        }

        let slidePattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 4
        let prefixPattern =
            try c.decodeIfPresent(Int.self, forKey: .prefixDenseSlidingWindowPattern) ?? 1
        let firstK = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        slidingWindowPattern = slidePattern
        prefixDenseSlidingWindowPattern = prefixPattern
        firstKDenseReplace = firstK

        if let explicitTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = explicitTypes
        } else {
            // HF: full_attention when (i + 1) is a multiple of the pattern. Prefix layers use
            // the prefix pattern, the rest use the sliding pattern.
            let count = try c.decode(Int.self, forKey: .hiddenLayers)
            layerTypes = (0 ..< count).map { i in
                let pattern = i < firstK ? prefixPattern : slidePattern
                return pattern > 0 && (i + 1) % pattern == 0 ? "full_attention" : "sliding_attention"
            }
        }

        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        prefixDenseIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: .prefixDenseIntermediateSize)

        let experts = try c.decodeIfPresent(Int.self, forKey: .numExperts)
        if let experts {
            numExperts = experts
        } else {
            // UNVERIFIED alias, kept from the earlier overlay. Not an HF key.
            numExperts = try c.decodeIfPresent(Int.self, forKey: .numLocalExperts)
        }
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 2
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        sharedExpertCombinationStrategy =
            try c.decodeIfPresent(String.self, forKey: .sharedExpertCombinationStrategy)
            ?? "average"
        expertSelectionFn =
            try c.decodeIfPresent(String.self, forKey: .expertSelectionFn) ?? "softmax"
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
    }

    public func validateModelConfiguration() throws {
        let dims = [
            headDim, hiddenSize, hiddenLayers, intermediateSize, attentionHeads, kvHeads,
            vocabularySize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: head, hidden, layer, intermediate and vocab sizes must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ $0 == "sliding_attention" || $0 == "full_attention" })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: layer_types needs one sliding_attention or full_attention per layer")
        }
        guard (0 ... hiddenLayers).contains(firstKDenseReplace) else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: first_k_dense_replace must be between 0 and num_hidden_layers")
        }
        guard slidingWindowPattern > 0, prefixDenseSlidingWindowPattern > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: sliding window patterns must be positive")
        }
        if let numExperts {
            guard numExperts > 0, numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw ModelFactoryError.invalidConfiguration(
                    "cohere2_moe: need 0 < num_experts_per_tok <= num_experts")
            }
        }
        guard numSharedExperts >= 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: num_shared_experts must not be negative")
        }
        guard ["sum", "average"].contains(sharedExpertCombinationStrategy) else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: shared_expert_combination_strategy must be sum or average")
        }
        guard ["softmax", "sigmoid"].contains(expertSelectionFn) else {
            throw ModelFactoryError.invalidConfiguration(
                "cohere2_moe: expert_selection_fn must be softmax or sigmoid")
        }
    }
}
