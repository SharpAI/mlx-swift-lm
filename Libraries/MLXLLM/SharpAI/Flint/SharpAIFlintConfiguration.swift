// SharpAI fork overlay: flint config. Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow vllm/models/flint/configs.py (FlintConfig).
// Covers the BF16 (safetensors) layout only. The vLLM NVFP4 GGUF path and its kernels are not ported.

import Foundation
import MLXLMCommon

enum SharpAIFlintError: Error, Equatable {
    case invalidConfiguration(String)
}

public struct SharpAIFlintConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var hiddenAct: String
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeParameters: [String: StringOrNumber]?
    var attentionBias: Bool
    var useSlidingWindow: Bool
    var slidingWindow: Int?
    var layerTypes: [String]
    var noRopeLayerIndices: [Int]
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var numExpertsPerTok: Int
    var numExperts: Int
    var normTopkProb: Bool
    var routerScoreFunction: String
    var mlpOnlyLayers: [Int]

    /// Sliding window used by sliding layers. Nil when sliding windows are disabled.
    var effectiveSlidingWindow: Int? {
        useSlidingWindow ? slidingWindow : nil
    }

    /// `rope_parameters.rope_theta` when present, else the top-level `rope_theta`.
    var resolvedRopeTheta: Float {
        ropeParameters?["rope_theta"]?.asFloat() ?? ropeTheta
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_act"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case noRopeLayerIndices = "no_rope_layer_indices"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case numExpertsPerTok = "num_experts_per_tok"
        case numExperts = "num_experts"
        case normTopkProb = "norm_topk_prob"
        case routerScoreFunction = "router_score_function"
        case mlpOnlyLayers = "mlp_only_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "flint"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 150_144
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        let layers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 49
        hiddenLayers = layers
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        useSlidingWindow = try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? true
        // JSON null is a valid value here (no window), so only an absent key takes the default.
        if c.contains(.slidingWindow) {
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        } else {
            slidingWindow = 513
        }
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: layers)
        noRopeLayerIndices =
            try c.decodeIfPresent([Int].self, forKey: .noRopeLayerIndices) ?? []
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
        sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 1024
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 6
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routerScoreFunction =
            try c.decodeIfPresent(String.self, forKey: .routerScoreFunction) ?? "sigmoid"
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
    }
}

extension SharpAIFlintConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard hiddenAct == "silu" else {
            throw SharpAIFlintError.invalidConfiguration("unsupported hidden_act \(hiddenAct)")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIFlintError.invalidConfiguration(
                "layer_types has \(layerTypes.count) entries but num_hidden_layers is \(hiddenLayers)")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIFlintError.invalidConfiguration(
                "num_attention_heads must be a positive multiple of num_key_value_heads")
        }
        if let ropeType = ropeParameters?["rope_type"] ?? ropeParameters?["type"],
            ropeType != .string("default")
        {
            throw SharpAIFlintError.invalidConfiguration(
                "unsupported rope type in rope_parameters; only default RoPE is implemented")
        }
        guard routerScoreFunction == "sigmoid" || routerScoreFunction == "softmax" else {
            throw SharpAIFlintError.invalidConfiguration(
                "unsupported router_score_function \(routerScoreFunction)")
        }
        if mlpOnlyLayers.count < hiddenLayers {
            guard numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw SharpAIFlintError.invalidConfiguration(
                    "num_experts_per_tok must be in 1...num_experts")
            }
        }
    }
}
