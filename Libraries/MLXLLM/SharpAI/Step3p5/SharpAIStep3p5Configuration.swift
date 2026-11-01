//
//  SharpAIStep3p5Configuration.swift
//  MLXLLM
//
//  Text config for `step3p5` (StepFun Step-3.5-Flash, Step3p5ForCausalLM).
//  Field names follow scripts/conformance/offline/fixtures/step3p5/stepfun-ai__Step-3.5-Flash.json.
//  Defaults for absent keys come from vLLM step3p5.py.
//  UNVERIFIED: no checkpoint weights were loaded or run.
//

import Foundation
import MLXLMCommon

enum SharpAIStep3p5Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "step3p5 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "step3p5 invalid configuration: \(message)"
        }
    }
}

/// A key that holds one value for all layers, or one value per layer.
private enum SharpAIStep3p5FloatList: Decodable {
    case list([Float])
    case scalar(Float)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let list = try? container.decode([Float].self) {
            self = .list(list)
        } else {
            self = .scalar(try container.decode(Float.self))
        }
    }
}

/// `attention_other_setting`: the head count used by the other attention type.
struct SharpAIStep3p5AttentionOtherSetting: Decodable, Sendable {
    var attentionType: String?
    var numAttentionHeads: Int?

    enum CodingKeys: String, CodingKey {
        case attentionType = "attention_type"
        case numAttentionHeads = "num_attention_heads"
    }
}

struct SharpAIStep3p5Configuration: Decodable, Sendable {
    var modelType: String
    var architectures: [String]
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var attentionGroups: Int
    var headDim: Int
    var numSlidingAttentionHeads: Int?
    var attentionOtherSetting: SharpAIStep3p5AttentionOtherSetting?
    var attentionBias: Bool
    var useHeadWiseAttnGate: Bool
    var attImplType: String
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var hiddenAct: String
    var tieWordEmbeddings: Bool

    // Attention layout. Per-layer lists cover the MTP layers too.
    var layerTypes: [String]
    var slidingWindow: Int?
    var useRopeLayers: [Bool]
    var ropeThetas: [Float]
    var partialRotaryFactors: [Float]
    var ropeScaling: [String: StringOrNumber]?
    var yarnOnlyTypes: [String]?
    var nestedRopeParameters: [String: [String: StringOrNumber]]?

    // MLP and MoE layout.
    var mlpLayerTypes: [String]?
    var moeLayersEnum: String?
    var moeNumExperts: Int
    var moeTopK: Int
    var moeIntermediateSize: Int
    var shareExpertDim: Int
    var useMoeRouterBias: Bool
    var needFp32Gate: Bool
    var moeRouterActivation: String
    var moeRouterScalingFactor: Float
    var normExpertWeight: Bool
    var swigluLimits: [Float]
    var swigluLimitsShared: [Float]

    /// MTP draft layers. The target model does not run them.
    var numNextnPredictLayers: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDim = "head_dim"
        case numSlidingAttentionHeads = "num_sliding_attention_heads"
        case attentionOtherSetting = "attention_other_setting"
        case attentionBias = "attention_bias"
        case useHeadWiseAttnGate = "use_head_wise_attn_gate"
        case attImplType = "att_impl_type"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useRopeLayers = "use_rope_layers"
        case ropeTheta = "rope_theta"
        case partialRotaryFactors = "partial_rotary_factors"
        case ropeScaling = "rope_scaling"
        case yarnOnlyTypes = "yarn_only_types"
        case ropeParameters = "rope_parameters"
        case mlpLayerTypes = "mlp_layer_types"
        case moeLayersEnum = "moe_layers_enum"
        case moeNumExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDim = "share_expert_dim"
        case useMoeRouterBias = "use_moe_router_bias"
        case needFp32Gate = "need_fp32_gate"
        case moeRouterActivation = "moe_router_activation"
        case moeRouterScalingFactor = "moe_router_scaling_factor"
        case normExpertWeight = "norm_expert_weight"
        case swigluLimits = "swiglu_limits"
        case swigluLimitsShared = "swiglu_limits_shared"
        case numNextnPredictLayers = "num_nextn_predict_layers"
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p5"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let layers = try c.decode(Int.self, forKey: .hiddenLayers)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        hiddenSize = hidden
        hiddenLayers = layers
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        attentionGroups = try c.decode(Int.self, forKey: .attentionGroups)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hidden / max(heads, 1))
        numSlidingAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numSlidingAttentionHeads)
        attentionOtherSetting = try c.decodeIfPresent(
            SharpAIStep3p5AttentionOtherSetting.self, forKey: .attentionOtherSetting)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        useHeadWiseAttnGate = try c.decodeIfPresent(Bool.self, forKey: .useHeadWiseAttnGate) ?? false
        attImplType = try c.decodeIfPresent(String.self, forKey: .attImplType) ?? "GQA"
        // UNVERIFIED: vLLM reads rms_norm_eps without a default. 1e-5 is an assumption.
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: layers)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        useRopeLayers = try c.decodeIfPresent([Bool].self, forKey: .useRopeLayers) ?? []

        let theta = try c.decodeIfPresent(SharpAIStep3p5FloatList.self, forKey: .ropeTheta)
        ropeThetas = SharpAIStep3p5Configuration.expand(theta, count: layers, fallback: 10000)
        let partial = try c.decodeIfPresent(
            SharpAIStep3p5FloatList.self, forKey: .partialRotaryFactors)
        partialRotaryFactors = SharpAIStep3p5Configuration.expand(
            partial, count: layers, fallback: 1.0)

        ropeScaling = try? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        yarnOnlyTypes = try c.decodeIfPresent([String].self, forKey: .yarnOnlyTypes)
        nestedRopeParameters = try? c.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)

        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        moeLayersEnum = try c.decodeIfPresent(String.self, forKey: .moeLayersEnum)
        moeNumExperts = try c.decodeIfPresent(Int.self, forKey: .moeNumExperts) ?? 0
        moeTopK = try c.decodeIfPresent(Int.self, forKey: .moeTopK) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        shareExpertDim = try c.decodeIfPresent(Int.self, forKey: .shareExpertDim) ?? 0
        useMoeRouterBias = try c.decodeIfPresent(Bool.self, forKey: .useMoeRouterBias) ?? false
        needFp32Gate = try c.decodeIfPresent(Bool.self, forKey: .needFp32Gate) ?? false
        moeRouterActivation =
            try c.decodeIfPresent(String.self, forKey: .moeRouterActivation) ?? "sigmoid"
        moeRouterScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 1.0
        // UNVERIFIED: vLLM reads norm_expert_weight without a default. false is an assumption.
        normExpertWeight = try c.decodeIfPresent(Bool.self, forKey: .normExpertWeight) ?? false
        swigluLimits = try c.decodeIfPresent([Float].self, forKey: .swigluLimits) ?? []
        swigluLimitsShared =
            try c.decodeIfPresent([Float].self, forKey: .swigluLimitsShared) ?? []
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0
    }

    /// One value per layer. A scalar applies to every layer. A missing key uses the fallback.
    private static func expand(
        _ value: SharpAIStep3p5FloatList?, count: Int, fallback: Float
    ) -> [Float] {
        switch value {
        case .list(let values)?:
            return values
        case .scalar(let single)?:
            return Array(repeating: single, count: count)
        case nil:
            return Array(repeating: fallback, count: count)
        }
    }
}

extension SharpAIStep3p5Configuration {

    func isSlidingLayer(_ layer: Int) -> Bool {
        layerTypes[layer] == "sliding_attention"
    }

    func windowSize(layer: Int) -> Int? {
        isSlidingLayer(layer) ? slidingWindow : nil
    }

    /// Query heads for one layer. Sliding layers can use their own head count.
    func queryHeads(layer: Int) -> Int {
        guard isSlidingLayer(layer) else { return attentionHeads }
        if let numSlidingAttentionHeads, numSlidingAttentionHeads > 0 {
            return numSlidingAttentionHeads
        }
        if attentionOtherSetting?.attentionType == "sliding_attention",
            let heads = attentionOtherSetting?.numAttentionHeads
        {
            return heads
        }
        return attentionHeads
    }

    func moeLayerIndices() -> Set<Int> {
        guard let moeLayersEnum else { return [] }
        return Set(
            moeLayersEnum.split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            })
    }

    /// Mirrors vLLM `_is_moe_layer`: mlp_layer_types first, then moe_layers_enum.
    func isMoELayer(_ layer: Int) -> Bool {
        if let mlpLayerTypes {
            return layer < mlpLayerTypes.count && mlpLayerTypes[layer] == "sparse"
        }
        if moeLayersEnum != nil {
            return moeLayerIndices().contains(layer)
        }
        return 0 < layer && layer < hiddenLayers
    }

    /// Rope settings for one layer, resolved as vLLM `_get_rope_parameters` does.
    func ropeParameters(layer: Int) -> [String: StringOrNumber] {
        let type = layerType(layer)
        if let nested = nestedRopeParameters?[type] {
            var rope = nested
            if rope["partial_rotary_factor"] == nil {
                rope["partial_rotary_factor"] = .float(1.0)
            }
            return rope
        }
        var rope = ropeScaling ?? [:]
        if let yarnOnlyTypes, !yarnOnlyTypes.contains(type) {
            rope = [:]
        }
        if rope["rope_type"] == nil, rope["type"] == nil {
            rope["rope_type"] = .string("default")
        }
        rope["rope_theta"] = .float(ropeThetas[layer])
        rope["partial_rotary_factor"] = .float(partialRotaryFactors[layer])
        return rope
    }

    func layerType(_ layer: Int) -> String {
        layerTypes[layer]
    }

    func ropeType(layer: Int) -> String {
        let rope = ropeParameters(layer: layer)
        if case .string(let name)? = rope["rope_type"] ?? rope["type"] {
            return name
        }
        return "default"
    }

    func ropeBase(layer: Int) -> Float {
        ropeParameters(layer: layer)["rope_theta"]?.asFloat() ?? 10000
    }

    func partialRotaryFactor(layer: Int) -> Float {
        ropeParameters(layer: layer)["partial_rotary_factor"]?.asFloat() ?? 1.0
    }

    /// Channels that get RoPE: `int(head_dim * partial_rotary_factor)`.
    func rotaryDimensions(layer: Int) -> Int {
        Int(Float(headDim) * partialRotaryFactor(layer: layer))
    }

    /// Routed-expert SwiGLU clamp. Zero means no clamp.
    func routedSwigluLimit(layer: Int) -> Float {
        swigluLimits.indices.contains(layer) ? swigluLimits[layer] : 0
    }

    /// Shared-expert and dense-MLP SwiGLU clamp. Zero means no clamp.
    func sharedSwigluLimit(layer: Int) -> Float {
        swigluLimitsShared.indices.contains(layer) ? swigluLimitsShared[layer] : 0
    }
}

extension SharpAIStep3p5Configuration: ModelConfigurationValidating {

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        guard modelType == "step3p5" else {
            throw SharpAIStep3p5Error.invalidConfiguration("model_type must be step3p5")
        }
        if architectures.contains("Step3p5MTP"), !architectures.contains("Step3p5ForCausalLM") {
            throw SharpAIStep3p5Error.unsupported(
                "Step3p5MTP is the draft head, not the target model")
        }
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0, intermediateSize > 0 else {
            throw SharpAIStep3p5Error.invalidConfiguration(
                "hidden_size, num_hidden_layers, vocab_size and intermediate_size must be > 0")
        }
        guard attImplType == "GQA" else {
            throw SharpAIStep3p5Error.unsupported("att_impl_type=\(attImplType)")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIStep3p5Error.unsupported("hidden_act=\(hiddenAct)")
        }
        guard attentionGroups > 0, headDim > 0, headDim % 2 == 0 else {
            throw SharpAIStep3p5Error.invalidConfiguration(
                "num_attention_groups must be > 0 and head_dim a positive even number")
        }
        guard layerTypes.count >= hiddenLayers else {
            throw SharpAIStep3p5Error.invalidConfiguration(
                "layer_types must have one entry per layer")
        }
        guard
            layerTypes.prefix(hiddenLayers).allSatisfy({
                $0 == "full_attention" || $0 == "sliding_attention"
            })
        else {
            throw SharpAIStep3p5Error.unsupported("layer_types with an unknown attention kind")
        }
        guard ropeThetas.count >= hiddenLayers, partialRotaryFactors.count >= hiddenLayers else {
            throw SharpAIStep3p5Error.invalidConfiguration(
                "rope_theta and partial_rotary_factors must cover every layer")
        }
        guard useRopeLayers.isEmpty || useRopeLayers.count >= hiddenLayers else {
            throw SharpAIStep3p5Error.invalidConfiguration(
                "use_rope_layers must cover every layer when set")
        }
        guard swigluLimits.allSatisfy({ $0 >= 0 }), swigluLimitsShared.allSatisfy({ $0 >= 0 })
        else {
            throw SharpAIStep3p5Error.invalidConfiguration("swiglu limits must be >= 0")
        }

        for layer in 0 ..< hiddenLayers {
            let heads = queryHeads(layer: layer)
            guard heads > 0, heads % attentionGroups == 0 else {
                throw SharpAIStep3p5Error.invalidConfiguration(
                    "num_attention_groups must divide every per-layer head count")
            }
            let kind = ropeType(layer: layer)
            guard kind == "default" || kind == "llama3" || kind == "yarn" else {
                throw SharpAIStep3p5Error.unsupported("rope_type=\(kind)")
            }
            let partial = partialRotaryFactor(layer: layer)
            guard partial == 1 || partial == 0.5 else {
                throw SharpAIStep3p5Error.unsupported("partial_rotary_factor=\(partial)")
            }
        }

        if (0 ..< hiddenLayers).contains(where: { self.isSlidingLayer($0) }) {
            guard let slidingWindow, slidingWindow > 0 else {
                throw SharpAIStep3p5Error.invalidConfiguration(
                    "sliding_window must be > 0 when sliding layers exist")
            }
        }

        // The checkpoint sets both flags. vLLM asserts them, so the overlay does too.
        guard useMoeRouterBias else {
            throw SharpAIStep3p5Error.unsupported("use_moe_router_bias must be true")
        }
        guard needFp32Gate else {
            throw SharpAIStep3p5Error.unsupported("need_fp32_gate must be true")
        }
        guard moeRouterActivation == "sigmoid" else {
            throw SharpAIStep3p5Error.unsupported("moe_router_activation=\(moeRouterActivation)")
        }

        let hasMoE = (0 ..< hiddenLayers).contains(where: { self.isMoELayer($0) })
        if hasMoE {
            guard moeNumExperts > 0, moeTopK > 0, moeTopK <= moeNumExperts else {
                throw SharpAIStep3p5Error.invalidConfiguration(
                    "moe_top_k must be in 1...moe_num_experts")
            }
            guard moeIntermediateSize > 0, shareExpertDim > 0 else {
                throw SharpAIStep3p5Error.invalidConfiguration(
                    "moe_intermediate_size and share_expert_dim must be > 0")
            }
        }
    }
}
