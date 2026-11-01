// SharpAI fork overlay: cohere_compass text decoder configuration. Not part of upstream mlx-swift-lm.
//
// Keys follow transformers `CohereCompassTextConfig`. The sample values come from the
// `text_config` block of CohereLabs/North-Micro-Vision-Instruct (fixture under
// scripts/conformance/offline/fixtures/cohere_compass). This decodes a flat text_config object only.
// The top-level config and vision_config are not decoded here.

import Foundation
import MLXLMCommon

public struct SharpAICohereCompassTextConfiguration: Decodable, Sendable {

    struct RopeParameters: Decodable, Sendable {
        let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let layerNormEps: Float
    public let logitScale: Float
    public let attentionBias: Bool
    public let slidingWindow: Int
    public let layerTypes: [String]

    /// RoPE base per layer, in `layerTypes` order. `nil` marks a NoPE layer, which is the case
    /// when `rope_parameters[layer_type]` is null (the `full_attention` entry in the reference).
    public let layerRopeTheta: [Float?]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
        case hiddenAct = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        self.hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        self.hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        self.kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        // Fallback is the reference rule: head_dim defaults to hidden_size // num_attention_heads.
        self.headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
        self.vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        self.layerNormEps = try c.decode(Float.self, forKey: .layerNormEps)
        // Reference rule: a text-only model uses scale 1.0 when logit_scale is None.
        self.logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? 1.0
        self.attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        self.slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        self.layerTypes = try c.decode([String].self, forKey: .layerTypes)

        let activation = try c.decode(String.self, forKey: .hiddenAct)
        guard activation == "silu" else {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenAct, in: c,
                debugDescription: "cohere_compass: only silu is supported, got \(activation)")
        }

        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: c,
                debugDescription:
                    "cohere_compass: layer_types has \(layerTypes.count) entries, "
                    + "expected num_hidden_layers=\(hiddenLayers)")
        }

        let ropeParameters = try c.decode(
            [String: RopeParameters?].self, forKey: .ropeParameters)
        self.layerRopeTheta = layerTypes.map { layerType -> Float? in
            guard let entry = ropeParameters[layerType] else { return nil }
            return entry?.ropeTheta
        }
    }
}
