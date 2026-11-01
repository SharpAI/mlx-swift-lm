//
//  SharpAIExaoneMoeConfiguration.swift
//  MLXLLM
//
//  Text config for `exaone_moe` (K-EXAONE). Field names follow the checkpoint
//  config.json, as read by vLLM's exaone_moe.py. Per-layer arrays are indexed by
//  the 0-based layer number.
//  UNVERIFIED: no real checkpoint config was loaded. Optional-field defaults are
//  marked inline.
//

import Foundation
import MLXLMCommon

enum SharpAIExaoneMoeError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "exaone_moe unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "exaone_moe invalid configuration: \(message)"
        }
    }
}

struct SharpAIExaoneMoeConfiguration: Decodable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var hiddenAct: String
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool

    // Attention layout.
    var layerTypes: [String]
    var slidingWindow: Int?
    var slidingWindows: [Int]?

    // MoE layout.
    var mlpLayerTypes: [String]?
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var numSharedExperts: Int
    var routedScalingFactor: Float
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int

    /// Per-layer SwiGLU clamp. A value <= 0 means no clamp for that layer.
    var swigluLimits: [Double]?

    // MTP head.
    var numNextnPredictLayers: Int
    var mtpLayerTypes: [String]?
    var mtpSlidingWindows: [Int]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case slidingWindows = "sliding_windows"
        case mlpLayerTypes = "mlp_layer_types"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case numSharedExperts = "num_shared_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case swigluLimits = "swiglu_limits"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case mtpLayerTypes = "mtp_layer_types"
        case mtpSlidingWindows = "mtp_sliding_windows"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "exaone_moe"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        // vLLM falls back to 1e6 when rope_theta is absent.
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false

        // vLLM reads layer_types directly. Absent means full attention everywhere here.
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        slidingWindows = try c.decodeIfPresent([Int].self, forKey: .slidingWindows)

        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        // UNVERIFIED default: vLLM reads norm_topk_prob directly, so true is a guess.
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1

        swigluLimits = try c.decodeIfPresent([Double].self, forKey: .swigluLimits)

        numNextnPredictLayers = try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 1
        mtpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mtpLayerTypes)
        mtpSlidingWindows = try c.decodeIfPresent([Int].self, forKey: .mtpSlidingWindows)
    }
}

extension SharpAIExaoneMoeConfiguration {

    /// Sliding window for a layer, or nil for full attention (vLLM `sliding_window_size`).
    /// MTP layers read the `mtp_*` arrays first and fall back to the main arrays.
    func windowSize(layer: Int, mtp: Bool) -> Int? {
        let types = mtp ? (mtpLayerTypes ?? layerTypes) : layerTypes
        guard layer < types.count, types[layer] != "full_attention" else { return nil }
        let windows = mtp ? (mtpSlidingWindows ?? slidingWindows) : slidingWindows
        if let windows, layer < windows.count {
            return windows[layer]
        }
        return slidingWindow
    }

    /// RoPE is off only on full-attention layers when sliding layers exist (NoPE),
    /// matching vLLM `use_rope`. MTP layers always use RoPE.
    func usesRope(layer: Int, mtp: Bool) -> Bool {
        let hasSlidingLayers = layerTypes.contains("sliding_attention")
        let window = windowSize(layer: layer, mtp: mtp) ?? 0
        return window > 0 || mtp || !hasSlidingLayers
    }

    /// SwiGLU clamp for a layer, or nil when the layer has no clamp.
    func swigluLimit(layer: Int) -> Float? {
        guard let swigluLimits, layer < swigluLimits.count, swigluLimits[layer] > 0 else {
            return nil
        }
        return Float(swigluLimits[layer])
    }

    /// True when the main-model layer uses a routed-expert (MoE) block.
    func isSparseLayer(_ layer: Int) -> Bool {
        guard let mlpLayerTypes, layer < mlpLayerTypes.count else { return false }
        return mlpLayerTypes[layer] == "sparse"
    }
}

extension SharpAIExaoneMoeConfiguration: ModelConfigurationValidating {

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        guard vocabularySize > 0, hiddenSize > 0, hiddenLayers > 0 else {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "vocab_size, hidden_size and num_hidden_layers must be > 0")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIExaoneMoeError.unsupported("hidden_act=\(hiddenAct)")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "num_key_value_heads must divide num_attention_heads")
        }
        guard headDim > 0, headDim % 2 == 0 else {
            throw SharpAIExaoneMoeError.invalidConfiguration("head_dim must be a positive even number")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "layer_types must have num_hidden_layers entries")
        }
        if let slidingWindows, slidingWindows.count != hiddenLayers {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "sliding_windows must have num_hidden_layers entries")
        }
        for layer in 0 ..< hiddenLayers where layerTypes[layer] != "full_attention" {
            guard let window = windowSize(layer: layer, mtp: false), window > 0 else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "sliding layer \(layer) has no positive sliding_window")
            }
        }

        if let mlpLayerTypes {
            guard mlpLayerTypes.count == hiddenLayers else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "mlp_layer_types must have num_hidden_layers entries")
            }
        }
        let hasSparseLayers = (0 ..< hiddenLayers).contains { isSparseLayer($0) }
        if numExperts > 0 && mlpLayerTypes == nil {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "mlp_layer_types is required when num_experts > 0")
        }
        if hasSparseLayers {
            guard numExperts > 0, numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "num_experts_per_tok must be in 1...num_experts")
            }
            guard moeIntermediateSize > 0, numSharedExperts >= 0 else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "moe_intermediate_size must be > 0 and num_shared_experts >= 0")
            }
            guard nGroup > 0, numExperts % nGroup == 0, topkGroup > 0, topkGroup <= nGroup else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "n_group must divide num_experts and topk_group must be in 1...n_group")
            }
            // The grouped router takes the top 2 scores inside each group.
            guard nGroup == 1 || numExperts / nGroup >= 2 else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "n_group > 1 requires at least 2 experts per group")
            }
        }

        if let swigluLimits, swigluLimits.count != hiddenLayers {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "swiglu_limits must have num_hidden_layers entries")
        }

        // MTP layers are indexed 0 ..< num_nextn_predict_layers.
        guard numNextnPredictLayers >= 0 else {
            throw SharpAIExaoneMoeError.invalidConfiguration(
                "num_nextn_predict_layers must be >= 0")
        }
        if numNextnPredictLayers > 0 {
            let mtpTypes = mtpLayerTypes ?? layerTypes
            guard numNextnPredictLayers <= mtpTypes.count else {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "mtp_layer_types must cover num_nextn_predict_layers")
            }
            if let mtpSlidingWindows, mtpSlidingWindows.count < numNextnPredictLayers {
                throw SharpAIExaoneMoeError.invalidConfiguration(
                    "mtp_sliding_windows must cover num_nextn_predict_layers")
            }
            for layer in 0 ..< numNextnPredictLayers where mtpTypes[layer] != "full_attention" {
                guard let window = windowSize(layer: layer, mtp: true), window > 0 else {
                    throw SharpAIExaoneMoeError.invalidConfiguration(
                        "MTP sliding layer \(layer) has no positive window")
                }
            }
        }
    }
}
