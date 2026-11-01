// SharpAI fork overlay: voxtral_realtime configuration. Not part of upstream mlx-swift-lm.
//
// Keys follow the Voxtral-Mini-4B-Realtime-2602 config.json. Missing keys fall back to the
// defaults here.
// UNVERIFIED: ada_rms_norm_t_cond_dim is not in the config. 32 is an assumed default.

import Foundation
import MLXLMCommon

public enum SharpAIVoxtralRealtimeError: LocalizedError {
    case unsupportedConfiguration(String)
    case missingAudioInput
    case missingWeight(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let reason):
            return "VoxtralRealtime: unsupported configuration: \(reason)."
        case .missingAudioInput:
            return "VoxtralRealtime: prepare needs audio features in LMInput.audio."
        case .missingWeight(let key):
            return "VoxtralRealtime: checkpoint is missing '\(key)'."
        }
    }
}

/// `rope_parameters` in either sub-config. Only the theta is read.
struct SharpAIVoxtralRealtimeRopeParameters: Decodable {
    let ropeTheta: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
    }
}

/// `audio_config`: a causal encoder with a conv stem, RoPE and sliding-window attention.
public struct SharpAIVoxtralRealtimeAudioConfiguration: Decodable, Sendable {
    public var hiddenSize: Int = 1280
    public var hiddenLayers: Int = 32
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 32
    public var headDim: Int = 64
    public var intermediateSize: Int = 5120
    public var hiddenAct: String = "silu"
    public var numMelBins: Int = 128
    public var maxPositionEmbeddings: Int = 1500
    public var slidingWindow: Int = 750
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 1_000_000

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case numMelBins = "num_mel_bins"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1500
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 750
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        let rope = try c.decodeIfPresent(
            SharpAIVoxtralRealtimeRopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 1_000_000
    }
}

/// `text_config`: a Llama-style decoder with GQA, RoPE and sliding-window attention.
public struct SharpAIVoxtralRealtimeTextConfiguration: Decodable, Sendable {
    public var hiddenSize: Int = 3072
    public var hiddenLayers: Int = 26
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 9216
    public var hiddenAct: String = "silu"
    public var vocabularySize: Int = 131072
    public var maxPositionEmbeddings: Int = 131072
    public var slidingWindow: Int = 8192
    public var rmsNormEps: Float = 1e-5
    public var ropeTheta: Float = 1_000_000
    public var tieWordEmbeddings: Bool = true

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9216
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131072
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 8192
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        let rope = try c.decodeIfPresent(
            SharpAIVoxtralRealtimeRopeParameters.self, forKey: .ropeParameters)
        ropeTheta = rope?.ropeTheta ?? 1_000_000
    }
}

/// Top-level `VoxtralRealtimeConfig`: audio tower, projector and text decoder.
public struct SharpAIVoxtralRealtimeConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public var modelType: String = "voxtral_realtime"
    public var audioConfiguration = SharpAIVoxtralRealtimeAudioConfiguration()
    public var textConfiguration = SharpAIVoxtralRealtimeTextConfiguration()
    /// Projector output width. Must equal the text hidden size.
    public var hiddenSize: Int = 3072
    public var projectorHiddenAct: String = "gelu"
    /// Read for config parity. The prompt alignment does not use it yet.
    public var audioLengthPerTok: Int = 8
    public var defaultNumDelayTokens: Int = 6
    /// Audio frames merged into one projector input.
    public var downsampleFactor: Int = 4
    /// Width of the bottleneck in the delay-conditioned ada norm.
    public var adaNormDimension: Int = 32

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioConfiguration = "audio_config"
        case textConfiguration = "text_config"
        case hiddenSize = "hidden_size"
        case projectorHiddenAct = "projector_hidden_act"
        case audioLengthPerTok = "audio_length_per_tok"
        case defaultNumDelayTokens = "default_num_delay_tokens"
        case downsampleFactor = "downsample_factor"
        case adaNormDimension = "ada_rms_norm_t_cond_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "voxtral_realtime"
        audioConfiguration =
            try c.decodeIfPresent(
                SharpAIVoxtralRealtimeAudioConfiguration.self, forKey: .audioConfiguration)
            ?? SharpAIVoxtralRealtimeAudioConfiguration()
        textConfiguration =
            try c.decodeIfPresent(
                SharpAIVoxtralRealtimeTextConfiguration.self, forKey: .textConfiguration)
            ?? SharpAIVoxtralRealtimeTextConfiguration()
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        projectorHiddenAct = try c.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        audioLengthPerTok = try c.decodeIfPresent(Int.self, forKey: .audioLengthPerTok) ?? 8
        defaultNumDelayTokens = try c.decodeIfPresent(Int.self, forKey: .defaultNumDelayTokens) ?? 6
        downsampleFactor = try c.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 4
        adaNormDimension = try c.decodeIfPresent(Int.self, forKey: .adaNormDimension) ?? 32
    }

    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ reason: String) throws {
            if !condition {
                throw SharpAIVoxtralRealtimeError.unsupportedConfiguration(reason)
            }
        }

        let audio = audioConfiguration
        let text = textConfiguration

        try require(audio.hiddenSize > 0, "audio hidden_size must be positive")
        try require(audio.hiddenLayers > 0, "audio num_hidden_layers must be positive")
        try require(audio.attentionHeads > 0, "audio num_attention_heads must be positive")
        try require(audio.headDim > 0 && audio.headDim % 2 == 0, "audio head_dim must be even")
        try require(
            audio.kvHeads == audio.attentionHeads,
            "audio encoder supports only full attention (kv heads equal heads)")
        try require(audio.numMelBins > 0, "num_mel_bins must be positive")
        try require(audio.slidingWindow > 0, "audio sliding_window must be positive")
        try require(audio.hiddenAct == "silu", "audio hidden_act must be silu")

        try require(text.hiddenSize > 0 && text.hiddenSize % 2 == 0, "text hidden_size must be even")
        try require(text.hiddenLayers > 0, "text num_hidden_layers must be positive")
        try require(text.attentionHeads > 0, "text num_attention_heads must be positive")
        try require(text.headDim > 0 && text.headDim % 2 == 0, "text head_dim must be even")
        try require(
            text.kvHeads > 0 && text.attentionHeads % text.kvHeads == 0,
            "text num_attention_heads must be a multiple of num_key_value_heads")
        try require(text.slidingWindow > 0, "text sliding_window must be positive")
        try require(text.hiddenAct == "silu", "text hidden_act must be silu")
        try require(text.tieWordEmbeddings, "only tied word embeddings are supported")

        try require(hiddenSize == text.hiddenSize, "hidden_size must equal text hidden_size")
        try require(downsampleFactor > 0, "downsample_factor must be positive")
        try require(projectorHiddenAct == "gelu", "projector_hidden_act must be gelu")
        try require(adaNormDimension > 0, "ada_rms_norm_t_cond_dim must be positive")
    }
}
