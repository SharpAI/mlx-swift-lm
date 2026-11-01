//
//  SharpAIVaultgemmaConfiguration.swift
//  MLXLLM
//
//  Text config for `vaultgemma` (VaultGemmaForCausalLM). Field names and defaults follow
//  transformers/models/vaultgemma/configuration_vaultgemma.py (Gemma 2 based).
//  UNVERIFIED: no checkpoint config.json was read. Absent keys take the HF source defaults.
//

import Foundation
import MLXLMCommon

enum SharpAIVaultgemmaError: Error, LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)

    var errorDescription: String? {
        switch self {
        case .unsupported(let message):
            return "vaultgemma unsupported: \(message)"
        case .invalidConfiguration(let message):
            return "vaultgemma invalid configuration: \(message)"
        }
    }
}

struct SharpAIVaultgemmaConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    var modelType: String
    var architectures: [String]
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDim: Int
    var hiddenAct: String
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeType: String
    var ropeTheta: Float
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var queryPreAttnScalar: Float

    /// Nil when the window is null or the layer is full attention. Nil means no window.
    var slidingWindow: Int?
    /// Per-layer `sliding_attention` or `full_attention`.
    var layerTypes: [String]
    /// Nil when the key is null in config.json. Absent keys use the HF default.
    var finalLogitSoftcapping: Float?
    var attnLogitSoftcapping: Float?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenAct = "hidden_activation"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case queryPreAttnScalar = "query_pre_attn_scalar"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attnLogitSoftcapping = "attn_logit_softcapping"
    }

    private struct RopeParameters: Decodable {
        let ropeType: String?
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeType = "rope_type"
            case ropeTheta = "rope_theta"
        }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let layerCount = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vaultgemma"
        architectures = try c.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 256_000
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2304
        hiddenLayers = layerCount
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9216
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        hiddenAct =
            try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6

        let rope = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        ropeType = rope?.ropeType ?? "default"
        ropeTheta =
            try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? rope?.ropeTheta ?? 10_000

        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        queryPreAttnScalar =
            try c.decodeIfPresent(Float.self, forKey: .queryPreAttnScalar) ?? 256

        slidingWindow = try Self.optionalValue(c, .slidingWindow, default: 4096)
        finalLogitSoftcapping = try Self.optionalValue(
            c, .finalLogitSoftcapping, default: Float(30.0))
        attnLogitSoftcapping = try Self.optionalValue(
            c, .attnLogitSoftcapping, default: Float(50.0))

        // HF default: even layers slide, odd layers are full attention.
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< layerCount).map { $0 % 2 == 0 ? "sliding_attention" : "full_attention" }
    }

    /// Reads a key that may be absent (use `value`) or null (nil).
    private static func optionalValue<T: Decodable>(
        _ container: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys, default value: T
    ) throws -> T? {
        guard container.contains(key) else { return value }
        return try container.decodeIfPresent(T.self, forKey: key)
    }

    func isSlidingLayer(_ index: Int) -> Bool {
        layerTypes[index] == "sliding_attention"
    }

    func validateModelConfiguration() throws {
        guard hiddenLayers > 0, vocabularySize > 0, hiddenSize > 0, intermediateSize > 0,
            headDim > 0
        else {
            throw SharpAIVaultgemmaError.invalidConfiguration(
                "layer, vocabulary, hidden, intermediate and head sizes must be positive")
        }
        guard attentionHeads > 0, kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIVaultgemmaError.invalidConfiguration(
                "num_attention_heads must be a positive multiple of num_key_value_heads")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIVaultgemmaError.invalidConfiguration(
                "layer_types has \(layerTypes.count) entries for \(hiddenLayers) layers")
        }
        for type in layerTypes where type != "sliding_attention" && type != "full_attention" {
            throw SharpAIVaultgemmaError.unsupported("layer type \(type)")
        }
        if let slidingWindow, slidingWindow <= 0 {
            throw SharpAIVaultgemmaError.invalidConfiguration("sliding_window must be positive")
        }
        guard ropeType == "default" else {
            throw SharpAIVaultgemmaError.unsupported("rope type \(ropeType)")
        }
        guard hiddenAct == "gelu_pytorch_tanh" || hiddenAct == "gelu" else {
            throw SharpAIVaultgemmaError.unsupported("hidden activation \(hiddenAct)")
        }
    }
}
