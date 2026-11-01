//
//  SharpAIStep5Configuration.swift
//  MLXLLM
//
//  Config for `step5` (Step5ForCausalLM, the text backbone of Step-5-Preview, about 598B
//  parameters). Field meanings and defaults follow the official reference:
//    - stepfun-ai/Step-5-Preview-BF16 config.json, text_config (sha256 a6f69688...)
//    - stepfun-ai/Step-5-Preview-BF16 inference/model.py (sha256 330a1668...)
//  The released step5_vl wrapper nests this block under text_config. A flat step5 config
//  (this struct's JSON shape) is accepted too.
//  Layer lists may carry the trailing next-token-prediction (MTP) entries; only the first
//  num_hidden_layers entries are kept.
//

import Foundation
import MLXLMCommon

enum SharpAIStep5Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "step5 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "step5 invalid configuration: \(message)"
        }
    }
}

struct SharpAIStep5Configuration: Decodable, Sendable, ModelConfigurationValidating {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    /// From num_key_value_heads, else num_attention_groups, else attentionHeads.
    var kvHeads: Int
    /// From head_dim. Derived as hidden_size / attentionHeads when absent.
    var headDim: Int
    /// True when head_dim is absent and headDim was derived from hidden_size.
    private(set) var headDimIsDerived: Bool
    var rmsNormEps: Float
    /// Scalar rope_theta, or the first entry when rope_theta is a per-layer list.
    var ropeTheta: Float
    /// RoPE base per main layer (length hiddenLayers).
    var ropeThetaPerLayer: [Float]
    /// Fraction of head_dim that is rotated, per main layer (length hiddenLayers).
    var partialRotaryFactors: [Float]
    var maxPositionEmbeddings: Int?
    var tieWordEmbeddings: Bool
    /// Activation name as written in config.json. The forward path uses the clamped SwiGLU.
    var hiddenAct: String?
    /// "sliding_attention" or "full_attention", per main layer (length hiddenLayers).
    var layerTypes: [String]
    /// Window size for sliding layers. Only applies to layers typed sliding_attention.
    var slidingWindow: Int?
    /// Factor of rope_scaling, or nil when absent. Only factor 1.0 is an identity and is accepted.
    var ropeScalingFactor: Float?
    var useHeadWiseAttnGate: Bool
    var useMoE: Bool
    /// Main layer indices that use the MoE block. All other layers use the dense MLP.
    var moeLayers: [Int]
    var moeIntermediateSize: Int
    var shareExpertDim: Int
    var moeNumExperts: Int
    var moeTopK: Int
    var normExpertWeight: Bool
    var moeRouterScalingFactor: Float
    /// Clamp limits of the routed/dense SwiGLU per layer. Empty means 7.0 everywhere.
    var swigluLimits: [Float]
    /// Clamp limits of the shared-expert SwiGLU per layer. Empty means 7.0 everywhere.
    var swigluLimitsShared: [Float]
    /// Whether DSA sparse attention is enabled. Its kernel is not ported.
    var sparseEnabled: Bool
    var sparseApplyToLayerTypes: [String]
    /// Whether the checkpoint carries a quantization_config (fp8 experts). Not ported.
    var hasQuantizationConfig: Bool

    var headsPerKVGroup: Int { kvHeads > 0 ? attentionHeads / kvHeads : 0 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case numAttentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case partialRotaryFactors = "partial_rotary_factors"
        case useHeadWiseAttnGate = "use_head_wise_attn_gate"
        case useMoE = "use_moe"
        case moeLayerList = "moe_layer_list"
        case moeLayersEnum = "moe_layers_enum"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDim = "share_expert_dim"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case numExpertsPerTok = "num_experts_per_tok"
        case normExpertWeight = "norm_expert_weight"
        case moeRouterScalingFactor = "moe_router_scaling_factor"
        case swigluLimits = "swiglu_limits"
        case swigluLimitsShared = "swiglu_limits_shared"
        case sparseConfig = "sparse_config"
        case quantizationConfig = "quantization_config"
    }

    private struct RopeScalingDescriptor: Decodable {
        let factor: Float?
    }

    private struct SparseDescriptor: Decodable {
        let enabled: Bool?
        let applyToLayerTypes: [String]?

        enum CodingKeys: String, CodingKey {
            case enabled
            case applyToLayerTypes = "apply_to_layer_types"
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step5"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)

        let explicitKVHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads)
        let attentionGroups = try c.decodeIfPresent(Int.self, forKey: .numAttentionGroups)
        kvHeads = explicitKVHeads ?? attentionGroups ?? attentionHeads

        if let explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .headDim) {
            headDim = explicitHeadDim
            headDimIsDerived = false
        } else {
            headDim = attentionHeads > 0 ? hiddenSize / attentionHeads : 0
            headDimIsDerived = true
        }

        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5

        if let thetaList = try? c.decode([Float].self, forKey: .ropeTheta) {
            ropeTheta = thetaList.first ?? 10_000
            ropeThetaPerLayer = Array(thetaList.prefix(hiddenLayers))
        } else {
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            ropeThetaPerLayer = Array(repeating: ropeTheta, count: hiddenLayers)
        }

        let factors = try c.decodeIfPresent([Float].self, forKey: .partialRotaryFactors) ?? []
        partialRotaryFactors =
            factors.isEmpty
            ? Array(repeating: 1.0, count: hiddenLayers)
            : Array(factors.prefix(hiddenLayers))

        let types = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        layerTypes =
            types.isEmpty
            ? Array(repeating: "full_attention", count: hiddenLayers)
            : Array(types.prefix(hiddenLayers))

        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)

        let scaling = try c.decodeIfPresent(RopeScalingDescriptor.self, forKey: .ropeScaling)
        ropeScalingFactor = scaling.map { $0.factor ?? 0 }

        useHeadWiseAttnGate =
            try c.decodeIfPresent(Bool.self, forKey: .useHeadWiseAttnGate) ?? true
        useMoE = try c.decodeIfPresent(Bool.self, forKey: .useMoE) ?? true

        if let list = try c.decodeIfPresent([Int].self, forKey: .moeLayerList) {
            moeLayers = list
        } else if let text = try c.decodeIfPresent(String.self, forKey: .moeLayersEnum) {
            moeLayers = text.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
        } else {
            moeLayers = []
        }

        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1536
        shareExpertDim = try c.decodeIfPresent(Int.self, forKey: .shareExpertDim) ?? 1536
        moeNumExperts = try c.decodeIfPresent(Int.self, forKey: .moeNumExperts) ?? 352
        let topK = try c.decodeIfPresent(Int.self, forKey: .moeTopK)
        let expertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeTopK = topK ?? expertsPerTok ?? 8
        normExpertWeight = try c.decodeIfPresent(Bool.self, forKey: .normExpertWeight) ?? true
        moeRouterScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 3.0
        swigluLimits = Array(
            (try c.decodeIfPresent([Float].self, forKey: .swigluLimits) ?? []).prefix(hiddenLayers))
        swigluLimitsShared = Array(
            (try c.decodeIfPresent([Float].self, forKey: .swigluLimitsShared) ?? [])
                .prefix(hiddenLayers))

        let sparse = try c.decodeIfPresent(SparseDescriptor.self, forKey: .sparseConfig)
        sparseEnabled = sparse.map { $0.enabled ?? true } ?? false
        sparseApplyToLayerTypes = sparse?.applyToLayerTypes ?? ["full_attention"]

        let hasQuantization = c.contains(.quantizationConfig)
        let quantizationIsNull = (try? c.decodeNil(forKey: .quantizationConfig)) ?? false
        hasQuantizationConfig = hasQuantization && !quantizationIsNull
    }

    // MARK: - Per-layer geometry

    func isSparseLayer(_ layer: Int) -> Bool {
        sparseEnabled && sparseApplyToLayerTypes.contains(layerTypes[layer])
    }

    func isMoELayer(_ layer: Int) -> Bool {
        useMoE && moeLayers.contains(layer)
    }

    /// The attention window for this layer, or nil for an unbounded (full) layer.
    func slidingWindowFor(_ layer: Int) -> Int? {
        layerTypes[layer] == "sliding_attention" ? slidingWindow : nil
    }

    func ropeBase(layer: Int) -> Float {
        ropeThetaPerLayer[layer]
    }

    /// Number of leading head dims that are rotated. Even, rounded down.
    func rotaryDimensions(layer: Int) -> Int {
        let span = Int((Float(headDim) * partialRotaryFactors[layer]).rounded())
        return span - span % 2
    }

    func swigluLimit(layer: Int) -> Float {
        swigluLimits.isEmpty ? 7.0 : swigluLimits[layer]
    }

    func sharedSwigluLimit(layer: Int) -> Float {
        swigluLimitsShared.isEmpty ? 7.0 : swigluLimitsShared[layer]
    }

    // MARK: - Validation

    public func validateModelConfiguration() throws {
        let positive: [(String, Int)] = [
            ("vocab_size", vocabularySize),
            ("hidden_size", hiddenSize),
            ("intermediate_size", intermediateSize),
            ("num_hidden_layers", hiddenLayers),
            ("num_attention_heads", attentionHeads),
            ("num_key_value_heads", kvHeads),
            ("head_dim", headDim),
        ]
        for (name, value) in positive where value <= 0 {
            throw SharpAIStep5Error.invalidConfiguration("\(name) must be positive, got \(value)")
        }
        if headDimIsDerived, hiddenSize % attentionHeads != 0 {
            throw SharpAIStep5Error.invalidConfiguration(
                "hidden_size \(hiddenSize) is not divisible by num_attention_heads \(attentionHeads)")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw SharpAIStep5Error.invalidConfiguration(
                "num_attention_heads \(attentionHeads) is not divisible by num_key_value_heads \(kvHeads)")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIStep5Error.invalidConfiguration(
                "layer_types has \(layerTypes.count) main-layer entries, expected \(hiddenLayers)")
        }
        for type in layerTypes where type != "sliding_attention" && type != "full_attention" {
            throw SharpAIStep5Error.invalidConfiguration("unknown layer type \(type)")
        }
        guard ropeThetaPerLayer.count == hiddenLayers else {
            throw SharpAIStep5Error.invalidConfiguration(
                "rope_theta has \(ropeThetaPerLayer.count) entries, expected \(hiddenLayers)")
        }
        guard partialRotaryFactors.count == hiddenLayers else {
            throw SharpAIStep5Error.invalidConfiguration(
                "partial_rotary_factors has \(partialRotaryFactors.count) entries, expected \(hiddenLayers)")
        }
        for layer in 0 ..< hiddenLayers {
            let span = rotaryDimensions(layer: layer)
            guard span >= 2, span <= headDim else {
                throw SharpAIStep5Error.invalidConfiguration(
                    "layer \(layer) rotary span \(span) is outside 2...\(headDim)")
            }
        }
        guard swigluLimits.isEmpty || swigluLimits.count == hiddenLayers else {
            throw SharpAIStep5Error.invalidConfiguration(
                "swiglu_limits has \(swigluLimits.count) entries, expected \(hiddenLayers)")
        }
        guard swigluLimitsShared.isEmpty || swigluLimitsShared.count == hiddenLayers else {
            throw SharpAIStep5Error.invalidConfiguration(
                "swiglu_limits_shared has \(swigluLimitsShared.count) entries, expected \(hiddenLayers)")
        }
        if useMoE && !moeLayers.isEmpty {
            for (name, value) in [
                ("moe_num_experts", moeNumExperts),
                ("moe_top_k", moeTopK),
                ("moe_intermediate_size", moeIntermediateSize),
                ("share_expert_dim", shareExpertDim),
            ] where value <= 0 {
                throw SharpAIStep5Error.invalidConfiguration("\(name) must be positive, got \(value)")
            }
            guard moeTopK <= moeNumExperts else {
                throw SharpAIStep5Error.invalidConfiguration(
                    "moe_top_k \(moeTopK) exceeds moe_num_experts \(moeNumExperts)")
            }
            for layer in moeLayers where layer < 0 || layer >= hiddenLayers {
                throw SharpAIStep5Error.invalidConfiguration(
                    "moe layer \(layer) is outside 0..<\(hiddenLayers)")
            }
        }
    }

    /// Rejects checkpoint features this port does not implement. Runs after
    /// validateModelConfiguration, before any weights are built.
    func validateSupportedArchitecture() throws {
        if hasQuantizationConfig {
            throw SharpAIStep5Error.unsupported(
                "quantized checkpoints (fp8 experts) are not ported; only BF16 is supported")
        }
        if let factor = ropeScalingFactor, factor != 1.0 {
            throw SharpAIStep5Error.unsupported(
                "rope_scaling factor \(factor) is not ported; only the identity factor 1.0 is")
        }
        for layer in 0 ..< hiddenLayers where layerTypes[layer] == "full_attention" {
            if isSparseLayer(layer) {
                throw SharpAIStep5Error.unsupported(
                    "layer \(layer) uses DSA sparse attention (inference/kernel.py), which is not ported yet")
            }
            throw SharpAIStep5Error.unsupported(
                "layer \(layer) is full attention without DSA; the reference has no dense full-attention path")
        }
    }
}
