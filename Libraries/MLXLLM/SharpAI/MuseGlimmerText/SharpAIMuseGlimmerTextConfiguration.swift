// SharpAI fork overlay: muse_glimmer_text (Muse-Glimmer text-only, MuseGlimmerForCausalLM).
// Not part of upstream mlx-swift-lm.
//
// The field set mirrors the text_config of the Muse-Glimmer VLM (MuseGlimmerTextConfiguration in
// Libraries/MLXVLM/Models/MuseGlimmer.swift). MLXLLM cannot import MLXVLM, so the type is repeated.

import Foundation
import MLXLMCommon

public struct SharpAIMuseGlimmerTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDimensions: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let attentionBias: Bool
    public let slidingWindow: Int
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool

    /// Per-layer attention kind: `sliding_attention` or `full_attention`.
    public let layerTypes: [String]

    /// Per-layer RoPE base. `0` marks a NoPE layer (no positional encoding).
    public let layerRopeTheta: [Float]

    public let ropeParameters: [String: StringOrNumber]?

    /// Fallback theta used to build the (unused) RoPE on NoPE layers.
    public var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 500_000.0 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "muse_glimmer_text"
        self.vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202048
        self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 6656
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 19968
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 52
        self.hiddenLayers = hiddenLayers
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        self.headDimensions =
            try container.decodeIfPresent(Int.self, forKey: .headDimensions) ?? 128
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        self.postNormEps = try container.decodeIfPresent(Float.self, forKey: .postNormEps) ?? 1e-8
        self.attentionBias =
            try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        self.qkScaleFactor =
            try container.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 3.87
        self.outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier)
            ?? 0.196_116_135_138_184_04
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 20.0
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        self.ropeParameters =
            try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

        // Full attention every 4th layer counting back from the last, so the final layer is full.
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            self.layerTypes = (0 ..< hiddenLayers).map { index in
                (hiddenLayers - 1 - index) % 4 == 0 ? "full_attention" : "sliding_attention"
            }
        }

        // Full-attention layers are NoPE (theta 0). Sliding layers use the base theta.
        if let decoded = try container.decodeIfPresent([Float].self, forKey: .layerRopeTheta) {
            self.layerRopeTheta = decoded
        } else {
            let baseTheta = self.ropeParameters?["rope_theta"]?.asFloat() ?? 500_000.0
            self.layerRopeTheta = self.layerTypes.map {
                $0 == "full_attention" ? 0 : baseTheta
            }
        }
    }
}

public enum SharpAIMuseGlimmerTextError: LocalizedError {
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message):
            "Invalid muse_glimmer_text configuration: \(message)"
        }
    }
}

extension SharpAIMuseGlimmerTextConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIMuseGlimmerTextError.invalidConfiguration(
                "layer_types has \(layerTypes.count) entries, expected \(hiddenLayers)")
        }
        guard layerRopeTheta.count == hiddenLayers else {
            throw SharpAIMuseGlimmerTextError.invalidConfiguration(
                "layer_rope_theta has \(layerRopeTheta.count) entries, expected \(hiddenLayers)")
        }
        for layerType in layerTypes
        where layerType != "full_attention" && layerType != "sliding_attention" {
            throw SharpAIMuseGlimmerTextError.invalidConfiguration(
                "unknown layer type '\(layerType)'")
        }
    }
}
