// SharpAI fork overlay: pangu_ultra_moe configuration. Not part of upstream mlx-swift-lm.
//
// Keys and defaults follow the openPangu-Ultra-MoE-718B sources: config.json of the
// openpangu/openPangu-Ultra-MoE-718B-V1.1 repo, and configuration_openpangu_moe.py
// (PanguUltraMoEConfig) of the FreedomIntelligence/openPangu-Ultra-MoE-718B repo, which is the
// only copy of that class on the Hub. The defaults used when a key is absent (norm_topk_prob,
// hidden_act, rope_theta) are the class defaults. Optional MoE and attention keys stay nil when
// absent, and validateForForward() rejects a config that omits one before any model is built.

import Foundation
import MLXLMCommon

public enum SharpAIPanguUltraMoeError: LocalizedError {
    /// No longer thrown by the creator, which builds the model. Kept so callers that match on it still compile.
    case forwardNotImplemented(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) has no forward pass."
        case .invalidConfiguration(let detail):
            return "PanguUltraMoe invalid configuration: \(detail)"
        }
    }
}

public struct SharpAIPanguUltraMoeConfiguration: Decodable, Sendable {
    public static let expectedModelType = "pangu_ultra_moe"
    public static let expectedArchitecture = "PanguUltraMoEForCausalLM"

    public let modelType: String
    public let architectures: [String]
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    /// Not read by the forward pass: the reference attention expands the latent KV to every head.
    public let kvHeads: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int?
    public let tieWordEmbeddings: Bool
    public let hiddenAct: String
    public let intermediateSize: Int?
    public let sandwichNorm: Bool

    // Mixture of experts. `firstDenseLayers` is num_dense_layers: layers below it use a dense MLP.
    public let routedExperts: Int?
    public let expertsPerToken: Int?
    public let moeIntermediateSize: Int?
    public let sharedExperts: Int?
    public let firstDenseLayers: Int?
    public let routedScalingFactor: Float?
    public let normTopkProb: Bool

    // Multi-head latent attention. A nil `attentionQLoraDim` selects a plain q_proj.
    public let attentionQLoraDim: Int?
    public let attentionKVLoraDim: Int?
    public let attentionQKDim: Int?
    public let attentionQKRopeDim: Int?
    public let attentionVDim: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
        case intermediateSize = "intermediate_size"
        case sandwichNorm = "sandwich_norm"
        case routedExperts = "num_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExperts = "num_shared_experts"
        case firstDenseLayers = "num_dense_layers"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopkProb = "norm_topk_prob"
        case attentionQLoraDim = "attention_q_lora_dim"
        case attentionKVLoraDim = "attention_kv_lora_dim"
        case attentionQKDim = "attention_qk_dim"
        case attentionQKRopeDim = "attention_qk_rope_dim"
        case attentionVDim = "attention_v_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.expectedModelType
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads =
            try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps =
            try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        // Class default in configuration_openpangu_moe.py.
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 25_600_000
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        // Class default in configuration_openpangu_moe.py; the published config.json omits it.
        hiddenAct =
            try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        // Read with getattr(config, "sandwich_norm", False) in modeling_openpangu_moe.py.
        sandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .sandwichNorm) ?? false
        routedExperts = try container.decodeIfPresent(Int.self, forKey: .routedExperts)
        expertsPerToken = try container.decodeIfPresent(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        sharedExperts = try container.decodeIfPresent(Int.self, forKey: .sharedExperts)
        firstDenseLayers =
            try container.decodeIfPresent(Int.self, forKey: .firstDenseLayers)
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
        // Class default in configuration_openpangu_moe.py; the published config.json omits it.
        normTopkProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        attentionQLoraDim =
            try container.decodeIfPresent(Int.self, forKey: .attentionQLoraDim)
        attentionKVLoraDim =
            try container.decodeIfPresent(Int.self, forKey: .attentionKVLoraDim)
        attentionQKDim = try container.decodeIfPresent(Int.self, forKey: .attentionQKDim)
        attentionQKRopeDim =
            try container.decodeIfPresent(Int.self, forKey: .attentionQKRopeDim)
        attentionVDim = try container.decodeIfPresent(Int.self, forKey: .attentionVDim)
    }

    /// Checks the model type, the architecture name when present, and the head and layer counts.
    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "model_type must be \(Self.expectedModelType)")
        }
        if !architectures.isEmpty, !architectures.contains(Self.expectedArchitecture) {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0
        else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "hidden size, layer and vocabulary counts must be positive, "
                    + "and attention heads must be divisible by kv heads")
        }
    }

    /// Checks everything the forward pass reads. Run it before constructing the model.
    public func validateForForward() throws {
        try validateModelConfiguration()

        var missing: [String] = []
        if intermediateSize == nil { missing.append("intermediate_size") }
        if routedExperts == nil { missing.append("num_routed_experts") }
        if expertsPerToken == nil { missing.append("num_experts_per_tok") }
        if moeIntermediateSize == nil { missing.append("moe_intermediate_size") }
        if firstDenseLayers == nil { missing.append("num_dense_layers") }
        if routedScalingFactor == nil { missing.append("routed_scaling_factor") }
        if attentionKVLoraDim == nil { missing.append("attention_kv_lora_dim") }
        if attentionQKDim == nil { missing.append("attention_qk_dim") }
        if attentionQKRopeDim == nil { missing.append("attention_qk_rope_dim") }
        if attentionVDim == nil { missing.append("attention_v_dim") }
        guard missing.isEmpty else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "forward pass needs " + missing.joined(separator: ", "))
        }

        guard hiddenAct == "silu" else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration("hidden_act must be silu")
        }
        guard let routedExperts, let expertsPerToken else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration("expert counts are missing")
        }
        guard routedExperts > 0, expertsPerToken > 0, expertsPerToken <= routedExperts else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "num_experts_per_tok must be between 1 and num_routed_experts")
        }
        guard let attentionKVLoraDim, let attentionQKDim, let attentionQKRopeDim,
            let attentionVDim, attentionQKRopeDim > 0, attentionKVLoraDim > 0,
            attentionQKDim > 0, attentionVDim > 0
        else {
            throw SharpAIPanguUltraMoeError.invalidConfiguration(
                "attention lora and head dimensions must be positive")
        }
    }
}
