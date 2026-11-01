// SharpAI fork overlay: moss_transcribe_diarize configuration. Not part of upstream mlx-swift-lm.
// UNVERIFIED: no checkpoint config.json is on disk. Field names follow a Whisper-style
// audio_config and a Qwen3 text_config. Defaults marked UNVERIFIED are guesses.

import Foundation
import MLXLMCommon

public enum SharpAIMossTranscribeDiarizeError: LocalizedError {
    case unsupported(String)
    case invalidInput(String)
    case audioTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "MOSS-Transcribe-Diarize unsupported configuration: \(detail)"
        case .invalidInput(let detail):
            return "MOSS-Transcribe-Diarize invalid input: \(detail)"
        case .audioTokenMismatch(let placeholders, let features):
            return
                "MOSS-Transcribe-Diarize has \(placeholders) audio tokens, \(features) embeddings."
        }
    }
}

/// `audio_config`: Whisper-style conv stem, learned positions, and pre-LN blocks.
public struct SharpAIMossTranscribeDiarizeAudioConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let numMelBins: Int
    public let maxSourcePositions: Int
    public let activationFunction: String
    /// Encoder frames stacked into one audio token. UNVERIFIED: 1 means no stacking.
    public let downsampleFactor: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case activationFunction = "activation_function"
        case downsampleFactor = "downsample_factor"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        // UNVERIFIED: 128 mel bins and 1500 positions are the Whisper defaults.
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions =
            try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        activationFunction =
            try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        downsampleFactor =
            try container.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 1
    }
}

/// `text_config` (Qwen3 decoder): consumes the audio embeddings.
public struct SharpAIMossTranscribeDiarizeTextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
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
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// Decode-only key for the nested RoPE block. It is kept out of `CodingKeys` so that
    /// synthesized `Encodable` only sees keys that map to stored properties.
    private enum ExtraKeys: String, CodingKey {
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
        let heads = try container.decode(Int.self, forKey: .attentionHeads)
        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenSize = hidden
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hidden / max(heads, 1))
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)

        let extraContainer = try decoder.container(keyedBy: ExtraKeys.self)
        let ropeParameters = try extraContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        ropeTheta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? 1_000_000
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Top-level `config.json` for `model_type: moss_transcribe_diarize`.
public struct SharpAIMossTranscribeDiarizeConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioConfiguration: SharpAIMossTranscribeDiarizeAudioConfiguration
    public let textConfiguration: SharpAIMossTranscribeDiarizeTextConfiguration
    /// UNVERIFIED: default is a placeholder id. The checkpoint value should win.
    public let audioTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfiguration = "audio_config"
        case textConfiguration = "text_config"
        case audioTokenId = "audio_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "moss_transcribe_diarize"
        audioConfiguration = try container.decode(
            SharpAIMossTranscribeDiarizeAudioConfiguration.self, forKey: .audioConfiguration)
        textConfiguration = try container.decode(
            SharpAIMossTranscribeDiarizeTextConfiguration.self, forKey: .textConfiguration)
        audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151_669
    }
}

extension SharpAIMossTranscribeDiarizeConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let audio = audioConfiguration
        let text = textConfiguration

        guard audio.activationFunction == "gelu" else {
            throw SharpAIMossTranscribeDiarizeError.unsupported(
                "audio activation_function \(audio.activationFunction)")
        }
        guard audio.numMelBins > 0, audio.maxSourcePositions > 0, audio.hiddenLayers > 0,
            audio.downsampleFactor > 0
        else {
            throw SharpAIMossTranscribeDiarizeError.unsupported(
                "audio encoder dimensions must be positive")
        }
        guard audio.attentionHeads > 0, audio.hiddenSize % audio.attentionHeads == 0 else {
            throw SharpAIMossTranscribeDiarizeError.unsupported(
                "audio hidden_size must be divisible by num_attention_heads")
        }
        guard text.hiddenLayers > 0, text.vocabularySize > 0, text.headDim > 0 else {
            throw SharpAIMossTranscribeDiarizeError.unsupported(
                "text decoder dimensions must be positive")
        }
        guard text.attentionHeads > 0, text.kvHeads > 0,
            text.attentionHeads % text.kvHeads == 0
        else {
            throw SharpAIMossTranscribeDiarizeError.unsupported(
                "text num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
