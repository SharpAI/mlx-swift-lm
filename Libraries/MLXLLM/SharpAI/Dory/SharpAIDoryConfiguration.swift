// SharpAI fork overlay: Dory configuration. Not part of upstream mlx-swift-lm.
//
// Field names and values come from scripts/conformance/offline/fixtures/dory/
// nvidia__NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16.json (tracked on this branch).
// UNVERIFIED: that file is not checked against the upstream checkpoint or a reference
// implementation, including ngpt_alpha_init and attention_output_gate_logit_scale_init.
// The forward pass is in SharpAIDoryModel.swift. See SharpAIDoryRegistration.swift.

import Foundation
import MLXLMCommon

public enum SharpAIDoryError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case forwardPassNotImplemented
}

/// Per-layer-type RoPE settings from `rope_parameters`.
public struct SharpAIDoryRopeParameters: Decodable, Sendable, Equatable {
    public let ropeType: String
    public let ropeTheta: Float
    public let partialRotaryFactor: Float?

    private enum CodingKeys: String, CodingKey {
        case ropeType = "rope_type"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }
}

public struct SharpAIDoryConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public let modelType: String
    public let architectures: [String]
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let hiddenAct: String
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let slidingWindow: Int?
    public let ngptAlphaInit: Float
    public let initializerRange: Float
    public let attentionOutputGateLogitScaleInit: Float?

    /// Sublayer counts. Input, recurrent and output stages together cover `hidden_layers`.
    public let nInputLayers: Int
    public let nRecurrentLayers: Int
    public let nOutputLayers: Int
    public let nRecurrentLoops: Int

    /// One entry per sublayer: "sliding_attention", "full_attention" or "mlp".
    public let layerTypes: [String]
    public let ropeParameters: [String: SharpAIDoryRopeParameters]

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenAct = "hidden_act"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case slidingWindow = "sliding_window"
        case ngptAlphaInit = "ngpt_alpha_init"
        case initializerRange = "initializer_range"
        case attentionOutputGateLogitScaleInit = "attention_output_gate_logit_scale_init"
        case nInputLayers = "n_input_layers"
        case nRecurrentLayers = "n_recurrent_layers"
        case nOutputLayers = "n_output_layers"
        case nRecurrentLoops = "n_recurrent_loops"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim = try container.decode(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        hiddenAct = try container.decode(String.self, forKey: .hiddenAct)
        tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)
        attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        mlpBias = try container.decode(Bool.self, forKey: .mlpBias)
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        ngptAlphaInit = try container.decode(Float.self, forKey: .ngptAlphaInit)
        initializerRange = try container.decode(Float.self, forKey: .initializerRange)
        attentionOutputGateLogitScaleInit = try container.decodeIfPresent(
            Float.self, forKey: .attentionOutputGateLogitScaleInit)
        nInputLayers = try container.decode(Int.self, forKey: .nInputLayers)
        nRecurrentLayers = try container.decode(Int.self, forKey: .nRecurrentLayers)
        nOutputLayers = try container.decode(Int.self, forKey: .nOutputLayers)
        nRecurrentLoops = try container.decode(Int.self, forKey: .nRecurrentLoops)
        layerTypes = try container.decode([String].self, forKey: .layerTypes)
        ropeParameters = try container.decode(
            [String: SharpAIDoryRopeParameters].self, forKey: .ropeParameters)
    }

    public func validateModelConfiguration() throws {
        guard nInputLayers + nRecurrentLayers + nOutputLayers == hiddenLayers else {
            throw SharpAIDoryError.invalidConfiguration(
                "input + recurrent + output layers must equal num_hidden_layers (\(hiddenLayers))")
        }
        guard layerTypes.count == hiddenLayers else {
            throw SharpAIDoryError.invalidConfiguration(
                "layer_types has \(layerTypes.count) entries, expected \(hiddenLayers)")
        }
        let known: Set<String> = ["sliding_attention", "full_attention", "mlp"]
        if let unknown = layerTypes.first(where: { !known.contains($0) }) {
            throw SharpAIDoryError.invalidConfiguration("unknown layer type \"\(unknown)\"")
        }
        guard nRecurrentLoops >= 1 else {
            throw SharpAIDoryError.invalidConfiguration("n_recurrent_loops must be at least 1")
        }
        // vLLM dory.py hard-codes SiLU and divides by initializer_range, so both are required here.
        guard hiddenAct == "silu" else {
            throw SharpAIDoryError.invalidConfiguration(
                "hidden_act \"\(hiddenAct)\" unsupported; the reference uses silu")
        }
        guard initializerRange > 0 else {
            throw SharpAIDoryError.invalidConfiguration("initializer_range must be positive")
        }
        // Tied embeddings are not covered by the reference read so far. lm_head source is unverified.
        guard !tieWordEmbeddings else {
            throw SharpAIDoryError.invalidConfiguration(
                "tie_word_embeddings=true is unsupported: the tied lm_head source is unverified")
        }
        guard kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIDoryError.invalidConfiguration(
                "num_attention_heads (\(attentionHeads)) must be a multiple of num_key_value_heads (\(kvHeads))")
        }
        for type in Set(layerTypes) where type != "mlp" {
            guard let rope = ropeParameters[type] else {
                throw SharpAIDoryError.invalidConfiguration("rope_parameters has no \"\(type)\" entry")
            }
            // The reference builds get_rope(head_dim, ..., neox) with no scaling, so only "default" is supported.
            guard rope.ropeType == "default" else {
                throw SharpAIDoryError.invalidConfiguration(
                    "rope_type \"\(rope.ropeType)\" unsupported for \(type)")
            }
        }
        if layerTypes.contains("sliding_attention"), slidingWindow == nil {
            throw SharpAIDoryError.invalidConfiguration(
                "sliding_attention layers need sliding_window")
        }
    }
}
