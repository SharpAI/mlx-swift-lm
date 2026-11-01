// SharpAI fork overlay: audioflamingo3 configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

public enum SharpAIAudioflamingo3Error: LocalizedError {
    case unsupported(String)
    case invalidInput(String)
    case audioTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "AudioFlamingo3 unsupported configuration: \(detail)"
        case .invalidInput(let detail):
            return "AudioFlamingo3 invalid input: \(detail)"
        case .audioTokenMismatch(let placeholders, let features):
            return "AudioFlamingo3 has \(placeholders) audio tokens, \(features) embeddings."
        }
    }
}

/// `audio_config` (`model_type: audioflamingo3_encoder`): Whisper-style encoder shape.
public struct SharpAIAudioflamingo3AudioConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let numMelBins: Int
    public let maxSourcePositions: Int
    public let activationFunction: String
    public let scaleEmbedding: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case activationFunction = "activation_function"
        case scaleEmbedding = "scale_embedding"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions =
            try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        activationFunction =
            try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        scaleEmbedding =
            try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
    }
}

/// `text_config` (`model_type: qwen2`): the Qwen2 decoder that consumes the audio embeddings.
public struct SharpAIAudioflamingo3TextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Decode-only keys for `rope_parameters`. Kept out of `CodingKeys` because it has no
    /// stored property, which would break the synthesized `Encodable` conformance.
    private enum ExtraCodingKeys: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    /// transformers v5 stores the RoPE base in `rope_parameters`.
    private struct RopeParameters: Decodable {
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let extras = try decoder.container(keyedBy: ExtraCodingKeys.self)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)

        let ropeParameters = try extras.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        let theta: Float =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? 1_000_000
        ropeTheta = theta
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Top-level `config.json` for `model_type: audioflamingo3`.
public struct SharpAIAudioflamingo3Configuration: Codable, Sendable {
    public let modelType: String
    public let audioConfiguration: SharpAIAudioflamingo3AudioConfiguration
    public let textConfiguration: SharpAIAudioflamingo3TextConfiguration
    public let audioTokenId: Int
    public let projectorBias: Bool
    public let projectorHiddenAct: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfiguration = "audio_config"
        case textConfiguration = "text_config"
        case audioTokenId = "audio_token_id"
        case projectorBias = "projector_bias"
        case projectorHiddenAct = "projector_hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "audioflamingo3"
        audioConfiguration = try container.decode(
            SharpAIAudioflamingo3AudioConfiguration.self, forKey: .audioConfiguration)
        textConfiguration = try container.decode(
            SharpAIAudioflamingo3TextConfiguration.self, forKey: .textConfiguration)
        audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151_669
        projectorBias = try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorHiddenAct =
            try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
    }
}

extension SharpAIAudioflamingo3Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let audio = audioConfiguration
        let text = textConfiguration

        guard audio.activationFunction == "gelu" else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "audio activation_function \(audio.activationFunction)")
        }
        guard !audio.scaleEmbedding else {
            throw SharpAIAudioflamingo3Error.unsupported("audio scale_embedding = true")
        }
        guard projectorHiddenAct == "gelu" else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "projector_hidden_act \(projectorHiddenAct)")
        }
        guard audio.numMelBins > 0, audio.maxSourcePositions > 0, audio.hiddenLayers > 0 else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "audio encoder dimensions must be positive")
        }
        guard audio.attentionHeads > 0, audio.hiddenSize % audio.attentionHeads == 0 else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "audio hidden_size must be divisible by num_attention_heads")
        }
        guard text.attentionHeads > 0, text.hiddenSize % text.attentionHeads == 0 else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "text hidden_size must be divisible by num_attention_heads")
        }
        guard text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIAudioflamingo3Error.unsupported(
                "text num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
