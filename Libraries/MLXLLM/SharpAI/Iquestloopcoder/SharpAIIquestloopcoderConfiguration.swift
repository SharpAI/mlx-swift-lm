//
//  SharpAIIquestloopcoderConfiguration.swift
//  MLXLLM
//
//  SharpAI fork overlay for `iquestloopcoder` (IQuestLoopCoderForCausalLM). Not upstream.
//
//  Keys and defaults follow the IQuestLab reference `configuration_iquestloopcoder.py`
//  (`__init__` defaults: rope_theta 500000, rms_norm_eps 1e-5, max_position_embeddings 8192,
//  loop_num 2, loop_window_size 64). The reference `__init__` defaults `head_dim` to 128;
//  this decoder defaults it to hidden_size / num_attention_heads instead, which matches
//  every real checkpoint (5120 / 40 = 128) and keeps tiny test configs consistent.
//

import Foundation
import MLXLMCommon

public enum SharpAIIquestloopcoderError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return "Model type \(type) is registered, but its forward pass is not available."
        }
    }
}

public struct SharpAIIquestloopcoderConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let tieWordEmbeddings: Bool
    /// Number of passes over the decoder stack (`loop_num`). Pass 0 is global attention.
    public let loopNum: Int
    /// Sliding window, in tokens, of the loop 1+ local key/value cache (`loop_window_size`).
    public let loopWindowSize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case loopNum = "loop_num"
        case loopWindowSize = "loop_window_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "iquestloopcoder"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        let heads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / max(heads, 1))
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000.0
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        loopNum = try c.decodeIfPresent(Int.self, forKey: .loopNum) ?? 2
        loopWindowSize = try c.decodeIfPresent(Int.self, forKey: .loopWindowSize) ?? 64
    }

    public func validateModelConfiguration() throws {
        let dims = [
            hiddenSize, hiddenLayers, attentionHeads, kvHeads, headDim, intermediateSize,
            vocabularySize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "iquestloopcoder: hidden, layer, head, head_dim, intermediate and vocab sizes must be positive")
        }
        guard attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "iquestloopcoder: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard loopNum >= 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "iquestloopcoder: loop_num must be at least 1")
        }
        guard loopWindowSize >= 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "iquestloopcoder: loop_window_size must be at least 1")
        }
    }
}
