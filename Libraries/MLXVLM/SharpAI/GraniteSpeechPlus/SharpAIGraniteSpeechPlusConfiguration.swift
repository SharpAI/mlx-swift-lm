// SharpAI fork overlay: granite_speech_plus configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow transformers `granite_speech` and `granite_speech_plus`:
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech/configuration_granite_speech.py
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/granite_speech_plus/modular_granite_speech_plus.py
// The projector defaults follow `Blip2QFormerConfig`:
// Verified against: https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/configuration_blip_2.py
// The released checkpoint (ibm-granite/granite-speech-4.1-2b-plus) was read as a summary only,
// not key by key, so its exact JSON layout is still unverified.

import Foundation
import MLXLMCommon

public enum SharpAIGraniteSpeechPlusError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)
    case audioTokenMismatch(tokens: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "GraniteSpeechPlus unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "GraniteSpeechPlus invalid configuration: \(detail)"
        case .audioTokenMismatch(let tokens, let features):
            return
                "GraniteSpeechPlus: \(tokens) audio placeholder tokens but \(features) audio embeddings"
        }
    }
}

/// `text_config`: the Granite decoder. Keys match `GraniteConfiguration`.
/// `rope_parameters` (transformers v5) becomes `ropeScaling`, so yarn settings are kept.
public struct SharpAIGraniteSpeechPlusTextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let logitsScaling: Float
    public let attentionMultiplier: Float
    public let embeddingMultiplier: Float
    public let residualMultiplier: Float
    public let maxPositionEmbeddings: Int
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case logitsScaling = "logits_scaling"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    /// `rope_parameters` is decoded only (transformers v5). It has no stored property, so it is
    /// kept out of `CodingKeys`; otherwise the synthesized `Encodable` conformance fails to compile.
    private enum RopeParametersKey: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let ropeContainer = try decoder.container(keyedBy: RopeParametersKey.self)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        logitsScaling = try container.decode(Float.self, forKey: .logitsScaling)
        attentionMultiplier = try container.decode(Float.self, forKey: .attentionMultiplier)
        embeddingMultiplier = try container.decode(Float.self, forKey: .embeddingMultiplier)
        residualMultiplier = try container.decode(Float.self, forKey: .residualMultiplier)
        maxPositionEmbeddings = try container.decode(Int.self, forKey: .maxPositionEmbeddings)
        attentionBias = try container.decode(Bool.self, forKey: .attentionBias)
        mlpBias = try container.decode(Bool.self, forKey: .mlpBias)
        tieWordEmbeddings = try container.decode(Bool.self, forKey: .tieWordEmbeddings)

        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        let legacyScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        if var parameters = try ropeContainer.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            ropeTheta = parameters["rope_theta"]?.asFloat() ?? topLevelTheta ?? 10_000_000
            parameters["rope_theta"] = nil
            ropeScaling = parameters
        } else {
            ropeTheta = topLevelTheta ?? 10_000_000
            ropeScaling = legacyScaling
        }
    }

    /// Checks head counts. Other fields are used as decoded.
    func validate() throws {
        guard hiddenLayers > 0, vocabularySize > 0 else {
            throw SharpAIGraniteSpeechPlusError.invalidConfiguration(
                "num_hidden_layers and vocab_size must be positive")
        }
        guard attentionHeads > 0, hiddenSize % attentionHeads == 0 else {
            throw SharpAIGraniteSpeechPlusError.invalidConfiguration(
                "hidden_size must divide by num_attention_heads")
        }
        guard kvHeads > 0, attentionHeads % kvHeads == 0 else {
            throw SharpAIGraniteSpeechPlusError.invalidConfiguration(
                "num_attention_heads must divide by num_key_value_heads")
        }
    }
}

/// `encoder_config`: the granite_speech_plus conformer (`GraniteSpeechPlusEncoderConfig`).
/// Defaults are the transformers defaults. Missing keys fall back to them.
public struct SharpAIGraniteSpeechPlusEncoderConfiguration: Decodable, Sendable {
    /// Mel-feature input width (`input_dim`).
    public var inputDim: Int = 160
    /// Number of conformer blocks (`num_layers`).
    public var numLayers: Int = 10
    /// Conformer width (`hidden_dim`).
    public var hiddenDim: Int = 1024
    public var feedforwardMult: Int = 4
    public var numHeads: Int = 8
    /// Per-head width. Nil means `hidden_dim / num_heads`.
    public var dimHead: Int? = nil
    /// Width of the Shaw relative-position bottleneck (`output_dim`), used by the mid-layer CTC
    /// self-conditioning.
    public var outputDim: Int = 42
    /// Block-attention window (`context_size`).
    public var contextSize: Int = 200
    /// Largest relative distance kept in the Shaw table (`max_pos_emb`).
    public var maxPosEmb: Int = 512
    public var convKernelSize: Int = 15
    public var convExpansionFactor: Int = 2
    /// Plus models only: 1-based block indices whose outputs are concatenated before the projector.
    /// Index 0 is the input projection. Nil means the plain encoder output.
    public var catHiddenLayers: [Int]? = nil

    public var resolvedDimHead: Int { dimHead ?? (hiddenDim / max(numHeads, 1)) }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case inputDim = "input_dim"
        case numLayers = "num_layers"
        case hiddenDim = "hidden_dim"
        case feedforwardMult = "feedforward_mult"
        case numHeads = "num_heads"
        case dimHead = "dim_head"
        case outputDim = "output_dim"
        case contextSize = "context_size"
        case maxPosEmb = "max_pos_emb"
        case convKernelSize = "conv_kernel_size"
        case convExpansionFactor = "conv_expansion_factor"
        case catHiddenLayers = "cat_hidden_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        inputDim = try c.decodeIfPresent(Int.self, forKey: .inputDim) ?? inputDim
        numLayers = try c.decodeIfPresent(Int.self, forKey: .numLayers) ?? numLayers
        hiddenDim = try c.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? hiddenDim
        feedforwardMult = try c.decodeIfPresent(Int.self, forKey: .feedforwardMult) ?? feedforwardMult
        numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? numHeads
        dimHead = try c.decodeIfPresent(Int.self, forKey: .dimHead)
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? outputDim
        contextSize = try c.decodeIfPresent(Int.self, forKey: .contextSize) ?? contextSize
        maxPosEmb = try c.decodeIfPresent(Int.self, forKey: .maxPosEmb) ?? maxPosEmb
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? convKernelSize
        convExpansionFactor =
            try c.decodeIfPresent(Int.self, forKey: .convExpansionFactor) ?? convExpansionFactor
        catHiddenLayers = try c.decodeIfPresent([Int].self, forKey: .catHiddenLayers)
    }

    /// Width of the encoder output, which the projector reads.
    public var outputWidth: Int {
        hiddenDim * ((catHiddenLayers?.count ?? 0) + 1)
    }
}

/// `projector_config`: the BLIP-2 Q-Former (`Blip2QFormerConfig`) used as the audio projector.
public struct SharpAIGraniteSpeechPlusProjectorConfiguration: Decodable, Sendable {
    /// Q-Former width (`hidden_size`).
    public var hiddenSize: Int = 768
    public var hiddenLayers: Int = 12
    public var attentionHeads: Int = 12
    public var intermediateSize: Int = 3072
    /// Width of the encoder states the cross-attention reads (`encoder_hidden_size`).
    public var encoderHiddenSize: Int = 1408
    public var layerNormEps: Float = 1e-12
    public var crossAttentionFrequency: Int = 2

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case encoderHiddenSize = "encoder_hidden_size"
        case layerNormEps = "layer_norm_eps"
        case crossAttentionFrequency = "cross_attention_frequency"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        encoderHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .encoderHiddenSize) ?? encoderHiddenSize
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        crossAttentionFrequency =
            try c.decodeIfPresent(Int.self, forKey: .crossAttentionFrequency)
            ?? crossAttentionFrequency
    }
}

/// Top-level `config.json` for `model_type: granite_speech_plus`.
public struct SharpAIGraniteSpeechPlusConfiguration: Decodable, Sendable {
    public let modelType: String
    public let textConfiguration: SharpAIGraniteSpeechPlusTextConfiguration
    public let encoderConfiguration: SharpAIGraniteSpeechPlusEncoderConfiguration
    public let projectorConfiguration: SharpAIGraniteSpeechPlusProjectorConfiguration
    /// Encoder frames per projector query (`downsample_rate`).
    public let downsampleRate: Int
    /// Encoder frames per Q-Former window (`window_size`).
    public let windowSize: Int
    /// Placeholder token id that audio embeddings replace (`audio_token_index`).
    public let audioTokenIndex: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case encoderConfiguration = "encoder_config"
        case projectorConfiguration = "projector_config"
        case downsampleRate = "downsample_rate"
        case windowSize = "window_size"
        case audioTokenIndex = "audio_token_index"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "granite_speech_plus"
        textConfiguration = try container.decode(
            SharpAIGraniteSpeechPlusTextConfiguration.self, forKey: .textConfiguration)
        encoderConfiguration =
            try container.decodeIfPresent(
                SharpAIGraniteSpeechPlusEncoderConfiguration.self, forKey: .encoderConfiguration)
            ?? SharpAIGraniteSpeechPlusEncoderConfiguration()
        projectorConfiguration =
            try container.decodeIfPresent(
                SharpAIGraniteSpeechPlusProjectorConfiguration.self, forKey: .projectorConfiguration)
            ?? SharpAIGraniteSpeechPlusProjectorConfiguration()
        downsampleRate = try container.decodeIfPresent(Int.self, forKey: .downsampleRate) ?? 5
        windowSize = try container.decodeIfPresent(Int.self, forKey: .windowSize) ?? 15
        audioTokenIndex = try container.decodeIfPresent(Int.self, forKey: .audioTokenIndex) ?? 49155
    }

    public func validateModelConfiguration() throws {
        try textConfiguration.validate()
        try validateEncoderAndProjector()
    }

    private func validateEncoderAndProjector() throws {
        let enc = encoderConfiguration
        let proj = projectorConfiguration
        let invalid = SharpAIGraniteSpeechPlusError.invalidConfiguration

        guard enc.inputDim > 0, enc.numLayers > 0, enc.hiddenDim > 0, enc.numHeads > 0 else {
            throw invalid("encoder input_dim, num_layers, hidden_dim and num_heads must be positive")
        }
        guard enc.hiddenDim % enc.numHeads == 0 || enc.dimHead != nil else {
            throw invalid("encoder hidden_dim must divide by num_heads when dim_head is unset")
        }
        guard enc.resolvedDimHead > 0 else {
            throw invalid("encoder dim_head must be positive")
        }
        guard enc.contextSize > 0, enc.contextSize <= enc.maxPosEmb else {
            throw invalid("encoder context_size must be in 1...max_pos_emb")
        }
        guard enc.convKernelSize > 0, enc.feedforwardMult > 0 else {
            throw invalid("encoder conv_kernel_size and feedforward_mult must be positive")
        }
        if let cat = enc.catHiddenLayers {
            for index in cat where index < 0 || index >= enc.numLayers {
                throw invalid("cat_hidden_layers index \(index) is out of range [0, \(enc.numLayers))")
            }
        }
        guard proj.encoderHiddenSize == enc.outputWidth else {
            throw invalid(
                "projector encoder_hidden_size \(proj.encoderHiddenSize) must equal the encoder "
                    + "output width \(enc.outputWidth)")
        }
        guard proj.hiddenSize > 0, proj.attentionHeads > 0,
            proj.hiddenSize % proj.attentionHeads == 0, proj.crossAttentionFrequency > 0
        else {
            throw invalid("projector hidden_size must divide by attention heads")
        }
        guard downsampleRate > 0, windowSize > 0, windowSize % downsampleRate == 0 else {
            throw invalid("window_size must be a positive multiple of downsample_rate")
        }
    }
}
