// SharpAI fork overlay: glmasr configuration. Not part of upstream mlx-swift-lm.
// Field names and defaults follow transformers' GlmAsrConfig and GlmAsrEncoderConfig
// (zai-org/GLM-ASR-Nano-2512). Unverified defaults are marked UNVERIFIED.

import Foundation
import MLXLMCommon

public enum SharpAIGlmasrError: LocalizedError {
    case unsupported(String)
    case invalidInput(String)
    case audioTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "GLM-ASR unsupported configuration: \(detail)"
        case .invalidInput(let detail):
            return "GLM-ASR invalid input: \(detail)"
        case .audioTokenMismatch(let placeholders, let features):
            return "GLM-ASR has \(placeholders) audio tokens, \(features) embeddings."
        }
    }
}

/// Lookup key for `rope_parameters`. It is kept out of `CodingKeys` so that the synthesized
/// `encode(to:)` does not look for a stored property that does not exist.
private struct RopeParametersKey: CodingKey {
    let stringValue: String

    init(_ stringValue: String) {
        self.stringValue = stringValue
    }

    init?(stringValue: String) {
        self.stringValue = stringValue
    }

    var intValue: Int? { nil }

    init?(intValue: Int) {
        return nil
    }

    static let ropeParameters = RopeParametersKey("rope_parameters")
}

/// `audio_config` (`model_type: glmasr_encoder`): Whisper-style conv stem, then partial-RoPE blocks.
public struct SharpAIGlmasrAudioConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let numMelBins: Int
    public let partialRotaryFactor: Float
    /// UNVERIFIED: the RoPE base of the audio encoder is not in the sources. Default 10000.
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numMelBins = "num_mel_bins"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public init(
        hiddenSize: Int = 1280,
        intermediateSize: Int = 5120,
        hiddenLayers: Int = 32,
        attentionHeads: Int = 20,
        kvHeads: Int? = nil,
        hiddenAct: String = "gelu",
        maxPositionEmbeddings: Int = 1500,
        numMelBins: Int = 128,
        partialRotaryFactor: Float = 0.5,
        ropeTheta: Float = 10_000
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numMelBins = numMelBins
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIGlmasrAudioConfiguration()
        let heads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? defaults.attentionHeads
        let ropeContainer = try decoder.container(keyedBy: RopeParametersKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        let theta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? defaults.ropeTheta

        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? defaults.hiddenSize,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? defaults.intermediateSize,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? defaults.hiddenLayers,
            attentionHeads: heads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads,
            hiddenAct: try container.decodeIfPresent(String.self, forKey: .hiddenAct)
                ?? defaults.hiddenAct,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self, forKey: .maxPositionEmbeddings) ?? defaults.maxPositionEmbeddings,
            numMelBins: try container.decodeIfPresent(Int.self, forKey: .numMelBins)
                ?? defaults.numMelBins,
            partialRotaryFactor: try container.decodeIfPresent(
                Float.self, forKey: .partialRotaryFactor) ?? defaults.partialRotaryFactor,
            ropeTheta: theta
        )
    }
}

/// `text_config` (`model_type: llama`): the Llama decoder that consumes the audio embeddings.
public struct SharpAIGlmasrTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public init(
        modelType: String = "llama",
        hiddenSize: Int = 2048,
        hiddenLayers: Int = 28,
        intermediateSize: Int = 6144,
        attentionHeads: Int = 16,
        kvHeads: Int? = nil,
        headDim: Int? = nil,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 59264,
        ropeTheta: Float = 10_000,
        attentionBias: Bool = false,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.headDim = headDim ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.attentionBias = attentionBias
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIGlmasrTextConfiguration()
        let heads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? defaults.attentionHeads
        let ropeContainer = try decoder.container(keyedBy: RopeParametersKey.self)
        let ropeParameters = try ropeContainer.decodeIfPresent(
            RopeParameters.self, forKey: .ropeParameters)
        let theta =
            try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeParameters?.ropeTheta
            ?? defaults.ropeTheta

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? defaults.modelType,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? defaults.hiddenSize,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? defaults.hiddenLayers,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? defaults.intermediateSize,
            attentionHeads: heads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? defaults.rmsNormEps,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? defaults.vocabularySize,
            ropeTheta: theta,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? defaults.attentionBias,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self, forKey: .tieWordEmbeddings) ?? defaults.tieWordEmbeddings
        )
    }
}

/// Top-level `config.json` for `model_type: glmasr`.
public struct SharpAIGlmasrConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioConfiguration: SharpAIGlmasrAudioConfiguration
    public let textConfiguration: SharpAIGlmasrTextConfiguration
    public let audioTokenId: Int
    public let projectorHiddenAct: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfiguration = "audio_config"
        case textConfiguration = "text_config"
        case audioTokenId = "audio_token_id"
        case projectorHiddenAct = "projector_hidden_act"
    }

    public init(
        modelType: String = "glmasr",
        audioConfiguration: SharpAIGlmasrAudioConfiguration = .init(),
        textConfiguration: SharpAIGlmasrTextConfiguration = .init(),
        audioTokenId: Int = 59260,
        projectorHiddenAct: String = "gelu"
    ) {
        self.modelType = modelType
        self.audioConfiguration = audioConfiguration
        self.textConfiguration = textConfiguration
        self.audioTokenId = audioTokenId
        self.projectorHiddenAct = projectorHiddenAct
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIGlmasrConfiguration()
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? defaults.modelType,
            audioConfiguration: try container.decodeIfPresent(
                SharpAIGlmasrAudioConfiguration.self, forKey: .audioConfiguration)
                ?? defaults.audioConfiguration,
            textConfiguration: try container.decodeIfPresent(
                SharpAIGlmasrTextConfiguration.self, forKey: .textConfiguration)
                ?? defaults.textConfiguration,
            audioTokenId: try container.decodeIfPresent(Int.self, forKey: .audioTokenId)
                ?? defaults.audioTokenId,
            projectorHiddenAct: try container.decodeIfPresent(
                String.self, forKey: .projectorHiddenAct) ?? defaults.projectorHiddenAct
        )
    }
}

extension SharpAIGlmasrConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let audio = audioConfiguration
        let text = textConfiguration

        guard modelType == "glmasr" else {
            throw SharpAIGlmasrError.unsupported("model_type \(modelType)")
        }
        guard audio.hiddenAct == "gelu" else {
            throw SharpAIGlmasrError.unsupported("audio hidden_act \(audio.hiddenAct)")
        }
        guard projectorHiddenAct == "gelu" else {
            throw SharpAIGlmasrError.unsupported("projector_hidden_act \(projectorHiddenAct)")
        }
        guard audio.hiddenLayers > 0, audio.numMelBins > 0 else {
            throw SharpAIGlmasrError.unsupported("audio layers and mel bins must be positive")
        }
        guard audio.attentionHeads > 0, audio.hiddenSize % audio.attentionHeads == 0 else {
            throw SharpAIGlmasrError.unsupported(
                "audio hidden_size must be divisible by num_attention_heads")
        }
        guard audio.kvHeads == audio.attentionHeads else {
            throw SharpAIGlmasrError.unsupported("audio grouped-query attention (kv heads)")
        }
        // The projector reads 4 stacked encoder frames, so intermediate_size must be 4 * hidden.
        guard audio.intermediateSize == 4 * audio.hiddenSize else {
            throw SharpAIGlmasrError.unsupported(
                "audio intermediate_size must equal 4 * hidden_size")
        }
        let headDim = audio.hiddenSize / audio.attentionHeads
        let rotaryDim = Int(Float(headDim) * audio.partialRotaryFactor)
        guard rotaryDim > 0, rotaryDim <= headDim, rotaryDim % 2 == 0 else {
            throw SharpAIGlmasrError.unsupported("audio rotary dimension \(rotaryDim)")
        }

        guard text.modelType == "llama" else {
            throw SharpAIGlmasrError.unsupported("text model_type \(text.modelType)")
        }
        guard text.attentionHeads > 0, text.kvHeads > 0, text.headDim > 0 else {
            throw SharpAIGlmasrError.unsupported("text head counts must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIGlmasrError.unsupported(
                "text num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
