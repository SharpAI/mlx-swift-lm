// SharpAI fork overlay: configs for the two architectures in the "unknown" entry. Not part of upstream.
//
// EagleCohereForCausalLM: EAGLE draft head for Cohere. Shares CohereConfiguration fields.
// PanguProMoEV2ForCausalLM: openPangu MoE. Key names follow vLLM's openpangu.py (UNVERIFIED
// against a real checkpoint: no config for either model exists in this repo).

import Foundation
import MLXLMCommon

public enum SharpAIUnknownError: LocalizedError {
    case forwardNotImplemented(String)
    case draftOnly(String)
    case unsupportedArchitecture(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let name):
            return "\(name) is registered, but its forward pass is not implemented yet."
        case .draftOnly(let name):
            return
                "\(name) is an EAGLE draft head. It needs the target model's hidden states "
                + "and cannot be loaded as a LanguageModel."
        case .unsupportedArchitecture(let names):
            return "No SharpAI overlay handles architectures: \(names)"
        case .invalidConfiguration(let detail):
            return "Invalid configuration: \(detail)"
        }
    }
}

public struct SharpAIEagleCohereConfiguration: Decodable, Sendable {
    public static let expectedArchitecture = "EagleCohereForCausalLM"

    public let architectures: [String]
    public let base: CohereConfiguration
    public let layerTypes: [String]?
    public let slidingWindow: Int?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case architectures
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        base = try CohereConfiguration(from: decoder)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }

    /// Checks the architecture name, the head and layer counts, and rejects sliding-window layers.
    public func validateModelConfiguration() throws {
        guard architectures.contains(Self.expectedArchitecture) else {
            throw SharpAIUnknownError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard base.hiddenSize > 0, base.hiddenLayers > 0, base.vocabularySize > 0,
            base.attentionHeads > 0, base.kvHeads > 0,
            base.attentionHeads % base.kvHeads == 0
        else {
            throw SharpAIUnknownError.invalidConfiguration(
                "hidden size, layer and vocabulary counts must be positive, "
                    + "and attention heads must be divisible by kv heads")
        }
        // Sliding-window (Cohere2) layers are not implemented. CohereTransformerBlock is dense only.
        if let layerTypes, layerTypes.contains(where: { $0 != "full_attention" }) {
            throw SharpAIUnknownError.unsupportedArchitecture(
                "\(Self.expectedArchitecture) with sliding-window layers (layer_types)")
        }
    }
}

public struct SharpAIPanguProMoEV2Configuration: Decodable, Sendable {
    public static let expectedArchitecture = "PanguProMoEV2ForCausalLM"

    public let architectures: [String]
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool
    public let intermediateSize: Int?

    // MoE keys (UNVERIFIED names, from vLLM openpangu.py).
    public let routedExperts: Int
    public let expertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExperts: Int?
    public let firstDenseLayers: Int
    public let routedScalingFactor: Float
    public let normTopKProb: Bool
    public let sandwichNorm: Bool
    public let paramSinkNumber: Int

    // MLA keys. Present only in the MLA attention variant (UNVERIFIED).
    public let qLoraRank: Int?
    public let kvLoraRank: Int?
    public let qkNopeHeadDim: Int?
    public let qkRopeHeadDim: Int?
    public let vHeadDim: Int?

    enum CodingKeys: String, CodingKey {
        case architectures
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
        case intermediateSize = "intermediate_size"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExperts = "n_shared_experts"
        case firstDenseLayers = "first_k_dense_replace"
        case routedScalingFactor = "routed_scaling_factor"
        case normTopKProb = "norm_topk_prob"
        case sandwichNorm = "sandwich_norm"
        case paramSinkNumber = "param_sink_number"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
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
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        routedExperts = try container.decode(Int.self, forKey: .routedExperts)
        expertsPerToken = try container.decode(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize = try container.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExperts = try container.decodeIfPresent(Int.self, forKey: .sharedExperts)
        firstDenseLayers =
            try container.decodeIfPresent(Int.self, forKey: .firstDenseLayers) ?? 0
        routedScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        normTopKProb =
            try container.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? false
        sandwichNorm =
            try container.decodeIfPresent(Bool.self, forKey: .sandwichNorm) ?? false
        paramSinkNumber =
            try container.decodeIfPresent(Int.self, forKey: .paramSinkNumber) ?? 0
        qLoraRank = try container.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try container.decodeIfPresent(Int.self, forKey: .kvLoraRank)
        qkNopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try container.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try container.decodeIfPresent(Int.self, forKey: .vHeadDim)
    }

    /// Checks the architecture name, counts, and the MoE routing fields.
    public func validateModelConfiguration() throws {
        guard architectures.contains(Self.expectedArchitecture) else {
            throw SharpAIUnknownError.invalidConfiguration(
                "architectures must include \(Self.expectedArchitecture)")
        }
        guard hiddenSize > 0, hiddenLayers > 0, vocabularySize > 0,
            attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0
        else {
            throw SharpAIUnknownError.invalidConfiguration(
                "hidden size, layer and vocabulary counts must be positive, "
                    + "and attention heads must be divisible by kv heads")
        }
        guard routedExperts > 0, expertsPerToken > 0, expertsPerToken <= routedExperts,
            moeIntermediateSize > 0
        else {
            throw SharpAIUnknownError.invalidConfiguration(
                "experts_per_token must be between 1 and n_routed_experts")
        }
        guard firstDenseLayers >= 0, firstDenseLayers <= hiddenLayers else {
            throw SharpAIUnknownError.invalidConfiguration(
                "first_k_dense_replace must be between 0 and num_hidden_layers")
        }
    }
}

/// Reads only the `architectures` list, used to pick an overlay entry.
struct SharpAIUnknownArchitectures: Decodable {
    let architectures: [String]

    enum CodingKeys: String, CodingKey {
        case architectures
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
    }
}
