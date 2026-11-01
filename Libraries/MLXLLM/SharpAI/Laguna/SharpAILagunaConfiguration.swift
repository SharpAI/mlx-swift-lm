//
//  SharpAILagunaConfiguration.swift
//  MLXLLM
//
//  Text config for `laguna` (Poolside Laguna-XS.2). Field names follow the checkpoint
//  config.json in scripts/conformance/offline/fixtures/laguna, as read by vLLM laguna.py.
//  Rope settings are resolved per layer type (full_attention, sliding_attention).
//  UNVERIFIED: no checkpoint weights were loaded. Defaults for absent keys are
//  assumptions from vLLM, not facts from the checkpoint.
//

import Foundation
import MLXLMCommon

enum SharpAILagunaError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "laguna unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "laguna invalid configuration: \(message)"
        }
    }
}

/// Per-head gating (`true` or `"per-head"`) or per-element gating (`"per-element"`).
enum SharpAILagunaGating: Equatable, Sendable {
    case off
    case perHead
    case perElement
    case unknown(String)
}

/// The checkpoint `gating` value is a bool or a mode string.
private enum SharpAILagunaGatingValue: Decodable {
    case flag(Bool)
    case mode(String)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = .flag(flag)
        } else {
            self = .mode(try container.decode(String.self))
        }
    }
}

struct SharpAILagunaConfiguration: Decodable, Sendable {
    var modelType: String
    var architectures: [String]
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var attentionHeadsPerLayer: [Int]
    var kvHeads: Int
    var headDim: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var hiddenAct: String
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var gating: SharpAILagunaGating
    var swaAttentionSinkEnabled: Bool

    // Attention layout. Resolved rope settings are keyed by layer type.
    var layerTypes: [String]
    var slidingWindow: Int?
    var ropeParameters: [String: [String: StringOrNumber]]

    // MoE layout.
    var mlpLayerTypes: [String]?
    var numExperts: Int
    var numExpertsPerTok: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var routedScalingFactor: Float
    var applyRouterWeightOnInput: Bool
    var routerLogitSoftcapping: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionHeadsPerLayer = "num_attention_heads_per_layer"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case gating
        case swaAttentionSinkEnabled = "swa_attention_sink_enabled"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeParameters = "rope_parameters"
        case swaRopeParameters = "swa_rope_parameters"
        case mlpLayerTypes = "mlp_layer_types"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case routedScalingFactor = "moe_routed_scaling_factor"
        case applyRouterWeightOnInput = "moe_apply_router_weight_on_input"
        case routerLogitSoftcapping = "moe_router_logit_softcapping"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "laguna"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let layers = try c.decode(Int.self, forKey: .hiddenLayers)
        hiddenSize = hidden
        hiddenLayers = layers
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)

        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionHeads = heads
        attentionHeadsPerLayer =
            try c.decodeIfPresent([Int].self, forKey: .attentionHeadsPerLayer)
            ?? Array(repeating: heads, count: layers)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / max(heads, 1))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        swaAttentionSinkEnabled =
            try c.decodeIfPresent(Bool.self, forKey: .swaAttentionSinkEnabled) ?? false

        let gatingValue = try c.decodeIfPresent(SharpAILagunaGatingValue.self, forKey: .gating)
        var mode = SharpAILagunaGating.off
        switch gatingValue {
        case .flag(let on)?:
            mode = on ? .perHead : .off
        case .mode(let name)?:
            switch name {
            case "per-head": mode = .perHead
            case "per-element": mode = .perElement
            default: mode = .unknown(name)
            }
        case nil:
            break
        }
        gating = mode

        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: layers)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)

        // Rope: nested per-layer-type form, or a flat dict shared by both layer types.
        let topTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let topPartial = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        let nestedRope = try? c.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        let flatRope = try? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        let swaRope = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .swaRopeParameters)

        let fullRope: [String: StringOrNumber]
        let slidingRope: [String: StringOrNumber]
        if let nestedRope {
            fullRope = nestedRope["full_attention"] ?? [:]
            slidingRope = nestedRope["sliding_attention"] ?? nestedRope["full_attention"] ?? [:]
        } else {
            fullRope = flatRope ?? [:]
            slidingRope = flatRope ?? [:]
        }
        ropeParameters = [
            "full_attention": SharpAILagunaConfiguration.resolveRope(
                fullRope, topTheta: topTheta, topPartial: topPartial),
            "sliding_attention": SharpAILagunaConfiguration.resolveRope(
                swaRope ?? slidingRope, topTheta: topTheta, topPartial: topPartial),
        ]

        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        applyRouterWeightOnInput =
            try c.decodeIfPresent(Bool.self, forKey: .applyRouterWeightOnInput) ?? false
        routerLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .routerLogitSoftcapping) ?? 0
    }

    /// Fills the top-level rope fallbacks into one layer type's rope dict, as vLLM does.
    private static func resolveRope(
        _ values: [String: StringOrNumber], topTheta: Float?, topPartial: Float?
    ) -> [String: StringOrNumber] {
        var out = values
        if out["rope_theta"] == nil, let topTheta {
            out["rope_theta"] = .float(topTheta)
        }
        if out["partial_rotary_factor"] == nil, let topPartial {
            out["partial_rotary_factor"] = .float(topPartial)
        }
        if out["rope_type"] == nil, out["type"] == nil {
            out["rope_type"] = .string("default")
        }
        // Keep attention_factor as is. YarnRoPE derives the same scale from factor.
        return out
    }
}

extension SharpAILagunaConfiguration {

    func isSlidingLayer(_ layer: Int) -> Bool {
        layerTypes[layer] == "sliding_attention"
    }

    func windowSize(layer: Int) -> Int? {
        isSlidingLayer(layer) ? slidingWindow : nil
    }

    func attentionHeads(layer: Int) -> Int {
        attentionHeadsPerLayer[layer]
    }

    func isSparseLayer(_ layer: Int) -> Bool {
        numExperts > 0 && mlpLayerTypes?[layer] == "sparse"
    }

    /// Resolved rope dict for this layer's type. Passed to `initializeRope` as the scaling config.
    func ropeScaling(layer: Int) -> [String: StringOrNumber] {
        ropeParameters[isSlidingLayer(layer) ? "sliding_attention" : "full_attention"] ?? [:]
    }

    func ropeTheta(layer: Int) -> Float? {
        ropeScaling(layer: layer)["rope_theta"]?.asFloat()
    }

    func ropeType(layer: Int) -> String {
        let rope = ropeScaling(layer: layer)
        if case .string(let name)? = rope["rope_type"] ?? rope["type"] {
            return name
        }
        return "default"
    }

    /// Channels that get RoPE: `int(head_dim * partial_rotary_factor)`.
    func rotaryDimensions(layer: Int) -> Int {
        let partial = ropeScaling(layer: layer)["partial_rotary_factor"]?.asFloat() ?? 1.0
        return Int(Float(headDim) * partial)
    }
}

extension SharpAILagunaConfiguration: ModelConfigurationValidating {

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        // The DFlash drafter shares model_type "laguna" but is not the target model.
        guard !architectures.contains("DFlashLagunaForCausalLM") else {
            throw SharpAILagunaError.unsupported(
                "DFlashLagunaForCausalLM is a speculative drafter, not the target model")
        }
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0, intermediateSize > 0 else {
            throw SharpAILagunaError.invalidConfiguration(
                "hidden_size, num_hidden_layers, vocab_size and intermediate_size must be > 0")
        }
        guard hiddenAct == "silu" else {
            throw SharpAILagunaError.unsupported("hidden_act=\(hiddenAct)")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAILagunaError.invalidConfiguration(
                "layer_types must have one entry per layer")
        }
        guard
            layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" })
        else {
            throw SharpAILagunaError.unsupported("layer_types with an unknown attention kind")
        }
        guard let mlpLayerTypes, mlpLayerTypes.count == hiddenLayers else {
            throw SharpAILagunaError.invalidConfiguration(
                "mlp_layer_types must have one entry per layer")
        }
        guard mlpLayerTypes.allSatisfy({ $0 == "dense" || $0 == "sparse" }) else {
            throw SharpAILagunaError.unsupported("mlp_layer_types with an unknown MLP kind")
        }
        guard attentionHeadsPerLayer.count == hiddenLayers else {
            throw SharpAILagunaError.invalidConfiguration(
                "num_attention_heads_per_layer must have one entry per layer")
        }
        guard kvHeads > 0, headDim > 0, headDim % 2 == 0 else {
            throw SharpAILagunaError.invalidConfiguration(
                "num_key_value_heads must be > 0 and head_dim a positive even number")
        }
        for heads in attentionHeadsPerLayer {
            guard heads > 0, heads % kvHeads == 0 else {
                throw SharpAILagunaError.invalidConfiguration(
                    "num_key_value_heads must divide every per-layer head count")
            }
        }
        guard !swaAttentionSinkEnabled else {
            throw SharpAILagunaError.unsupported("swa_attention_sink_enabled=true")
        }
        if case .unknown(let name) = gating {
            throw SharpAILagunaError.unsupported("gating=\(name)")
        }
        guard !applyRouterWeightOnInput else {
            throw SharpAILagunaError.unsupported("moe_apply_router_weight_on_input=true")
        }
        guard routerLogitSoftcapping >= 0 else {
            throw SharpAILagunaError.invalidConfiguration("moe_router_logit_softcapping must be >= 0")
        }

        for layer in 0 ..< hiddenLayers {
            guard ropeTheta(layer: layer) != nil else {
                throw SharpAILagunaError.invalidConfiguration(
                    "rope_theta is missing for layer type \(layerTypes[layer])")
            }
            let kind = ropeType(layer: layer)
            guard kind == "default" || kind == "yarn" else {
                throw SharpAILagunaError.unsupported("rope_type=\(kind)")
            }
            let dims = rotaryDimensions(layer: layer)
            guard dims > 0, dims % 2 == 0, dims <= headDim else {
                throw SharpAILagunaError.invalidConfiguration(
                    "partial_rotary_factor must give an even rotary dim in 1...head_dim")
            }
        }

        if layerTypes.contains("sliding_attention") {
            guard let slidingWindow, slidingWindow > 0 else {
                throw SharpAILagunaError.invalidConfiguration(
                    "sliding_window must be > 0 when sliding layers exist")
            }
        }

        if mlpLayerTypes.contains("sparse") {
            guard numExperts > 0, numExpertsPerTok > 0, numExpertsPerTok <= numExperts else {
                throw SharpAILagunaError.invalidConfiguration(
                    "num_experts_per_tok must be in 1...num_experts")
            }
            guard moeIntermediateSize > 0 else {
                throw SharpAILagunaError.invalidConfiguration("moe_intermediate_size must be > 0")
            }
        }
        guard sharedExpertIntermediateSize >= 0 else {
            throw SharpAILagunaError.invalidConfiguration(
                "shared_expert_intermediate_size must be >= 0")
        }
    }
}
