//
//  SharpAIMimoV2Configuration.swift
//  MLXLLM
//
//  Text config for `mimo_v2` (MiMo-V2.5-Pro, MiMoV2ForCausalLM). Key names and
//  defaults follow vllm/model_executor/models/mimo_v2.py.
//  Verified against: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/configuration_mimo_v2.py
//  (head_dim and v_head_dim defaults, sliding-window fallbacks to full-attention values).
//  UNVERIFIED: the mapping of an integer moe_layer_freq. The HF file expands it to a boolean
//  list, which differs from the all-dense reading in vLLM used here. Absent keys otherwise
//  use vLLM getattr() fallbacks. No checkpoint config.json was loaded into a test.
//  Omni checkpoints (vision_config / audio_config) are rejected at decode.
//

import Foundation
import MLXLMCommon

enum SharpAIMimoV2Error: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "mimo_v2 unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "mimo_v2 invalid configuration: \(message)"
        }
    }
}

struct SharpAIMimoV2Configuration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int

    // Global (full attention) layers.
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var vHeadDim: Int

    // Sliding-window layers (hybrid_layer_pattern == 1).
    var swaAttentionHeads: Int
    var swaKvHeads: Int
    var swaHeadDim: Int
    var swaVHeadDim: Int
    var hybridLayerPattern: [Int]
    var slidingWindowSize: Int

    var ropeTheta: Float
    var swaRopeTheta: Float
    var partialRotaryFactor: Float
    var maxPositionEmbeddings: Int
    var layernormEpsilon: Float
    var hiddenAct: String
    var attentionBias: Bool
    var addSwaAttentionSinkBias: Bool
    var attentionValueScale: Float?
    var attentionProjectionLayout: String?
    var tieWordEmbeddings: Bool
    var numNextnPredictLayers: Int

    // MoE. An empty moeLayerFreq means every layer is dense.
    var moeLayerFreq: [Int]
    var nRoutedExperts: Int?
    var numExpertsPerTok: Int?
    var moeIntermediateSize: Int?
    var nSharedExperts: Int
    var normTopkProb: Bool
    var nGroup: Int
    var topkGroup: Int
    var routedScalingFactor: Float

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vHeadDim = "v_head_dim"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKvHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case swaVHeadDim = "swa_v_head_dim"
        case hybridLayerPattern = "hybrid_layer_pattern"
        case slidingWindowSize = "sliding_window_size"
        case ropeTheta = "rope_theta"
        case swaRopeTheta = "swa_rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layernormEpsilon = "layernorm_epsilon"
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case addSwaAttentionSinkBias = "add_swa_attention_sink_bias"
        case attentionValueScale = "attention_value_scale"
        case attentionProjectionLayout = "attention_projection_layout"
        case tieWordEmbeddings = "tie_word_embeddings"
        case numNextnPredictLayers = "num_nextn_predict_layers"
        case moeLayerFreq = "moe_layer_freq"
        case nRoutedExperts = "n_routed_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case nSharedExperts = "n_shared_experts"
        case normTopkProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
    }

    // Probe keys only. Kept out of CodingKeys so encode(to:) can still be synthesized.
    private enum OmniProbeKeys: String, CodingKey {
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
    }

    init(from decoder: Decoder) throws {
        let probe = try decoder.container(keyedBy: OmniProbeKeys.self)
        if probe.contains(.visionConfig) || probe.contains(.audioConfig) {
            throw SharpAIMimoV2Error.unsupported(
                "Omni checkpoints (vision/audio towers) are not implemented")
        }

        let c = try decoder.container(keyedBy: CodingKeys.self)

        let rope = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)
        let fullHeads = try c.decode(Int.self, forKey: .attentionHeads)
        let fullKVHeads = try c.decode(Int.self, forKey: .kvHeads)
        // head_dim defaults to hidden_size // num_attention_heads, and v_head_dim to head_dim
        // (configuration_mimo_v2.py).
        let headDimValue =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (fullHeads > 0 ? hidden / fullHeads : 0)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mimo_v2"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = hidden
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)

        attentionHeads = fullHeads
        kvHeads = fullKVHeads
        headDim = headDimValue
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? headDimValue

        // Sliding-window geometry falls back to the full-attention values when unset
        // (configuration_mimo_v2.py).
        swaAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .swaAttentionHeads) ?? fullHeads
        swaKvHeads = try c.decodeIfPresent(Int.self, forKey: .swaKvHeads) ?? fullKVHeads
        let swaHeadDimValue =
            try c.decodeIfPresent(Int.self, forKey: .swaHeadDim) ?? headDimValue
        swaHeadDim = swaHeadDimValue
        swaVHeadDim = try c.decodeIfPresent(Int.self, forKey: .swaVHeadDim) ?? swaHeadDimValue
        hybridLayerPattern = try c.decode([Int].self, forKey: .hybridLayerPattern)
        slidingWindowSize = try c.decode(Int.self, forKey: .slidingWindowSize)

        ropeTheta = rope
        swaRopeTheta = try c.decodeIfPresent(Float.self, forKey: .swaRopeTheta) ?? rope
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        layernormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layernormEpsilon) ?? 1e-5
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        addSwaAttentionSinkBias =
            try c.decodeIfPresent(Bool.self, forKey: .addSwaAttentionSinkBias) ?? false
        attentionValueScale = try c.decodeIfPresent(Float.self, forKey: .attentionValueScale)
        attentionProjectionLayout =
            try c.decodeIfPresent(String.self, forKey: .attentionProjectionLayout)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        numNextnPredictLayers =
            try c.decodeIfPresent(Int.self, forKey: .numNextnPredictLayers) ?? 0

        // vLLM reads a per-layer list. A scalar (or absent) value means no MoE layers.
        if let list = try? c.decode([Int].self, forKey: .moeLayerFreq) {
            moeLayerFreq = list
        } else if let flags = try? c.decode([Bool].self, forKey: .moeLayerFreq) {
            moeLayerFreq = flags.map { $0 ? 1 : 0 }
        } else {
            moeLayerFreq = []
        }
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
    }

    /// True for a layer that uses the sliding-window attention geometry.
    func isSlidingWindowLayer(_ index: Int) -> Bool {
        index < hybridLayerPattern.count && hybridLayerPattern[index] == 1
    }

    /// True for a layer that uses the sparse MoE block.
    func isMoELayer(_ index: Int) -> Bool {
        index < moeLayerFreq.count && moeLayerFreq[index] != 0
    }
}

extension SharpAIMimoV2Configuration: ModelConfigurationValidating {

    /// Rejects configs this overlay cannot run. Throws before any weight is built.
    func validateModelConfiguration() throws {
        guard vocabularySize > 0, hiddenSize > 0, hiddenLayers > 0 else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "vocab_size, hidden_size and num_hidden_layers must be > 0")
        }
        guard hybridLayerPattern.count == hiddenLayers,
            hybridLayerPattern.allSatisfy({ $0 == 0 || $0 == 1 })
        else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "hybrid_layer_pattern must hold 0 or 1 for each of num_hidden_layers entries")
        }
        guard moeLayerFreq.isEmpty || moeLayerFreq.count == hiddenLayers else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "moe_layer_freq must have num_hidden_layers entries")
        }
        guard headDim > 0, vHeadDim > 0, swaHeadDim > 0, swaVHeadDim > 0 else {
            throw SharpAIMimoV2Error.invalidConfiguration("head dims must be > 0")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard swaAttentionHeads > 0, swaKvHeads > 0, swaAttentionHeads % swaKvHeads == 0 else {
            throw SharpAIMimoV2Error.invalidConfiguration(
                "swa_num_attention_heads must be a multiple of swa_num_key_value_heads")
        }
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
            throw SharpAIMimoV2Error.invalidConfiguration("partial_rotary_factor must be in (0, 1]")
        }
        guard hiddenAct == "silu" else {
            throw SharpAIMimoV2Error.unsupported("hidden_act \(hiddenAct); only silu is supported")
        }
        guard nSharedExperts == 0 else {
            throw SharpAIMimoV2Error.unsupported(
                "n_shared_experts > 0; the vLLM mimo_v2 reference has no shared expert")
        }
        if hybridLayerPattern.contains(1) {
            guard slidingWindowSize > 0 else {
                throw SharpAIMimoV2Error.invalidConfiguration(
                    "sliding_window_size must be > 0 when sliding-window layers exist")
            }
        }
        if (0 ..< hiddenLayers).contains(where: { isMoELayer($0) }) {
            guard let experts = nRoutedExperts, experts > 0,
                let topK = numExpertsPerTok, topK > 0, topK <= experts,
                let moeSize = moeIntermediateSize, moeSize > 0,
                nGroup > 0, experts % nGroup == 0, topkGroup > 0, topkGroup <= nGroup
            else {
                throw SharpAIMimoV2Error.invalidConfiguration(
                    "MoE layers need n_routed_experts, num_experts_per_tok, "
                        + "moe_intermediate_size and valid n_group / topk_group")
            }
        }
    }
}
