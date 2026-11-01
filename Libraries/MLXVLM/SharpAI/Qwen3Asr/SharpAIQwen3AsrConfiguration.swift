// SharpAI fork overlay: qwen3_asr configuration. Not part of upstream mlx-swift-lm.
//
// Hub checkpoints nest the model under "thinker_config". Transformers flattens it.
// Both layouts decode. Defaults follow transformers' Qwen3ASR configs.

import Foundation
import MLXLMCommon

public enum SharpAIQwen3AsrError: LocalizedError {
    case unsupportedConfiguration(String)
    case unsupportedBatch(Int)
    case invalidInput(String)
    case audioTokenMismatch(placeholders: Int, encoderRows: Int)
    case unexpectedAudio
    case unsupportedTask(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedConfiguration(let reason):
            return "Qwen3Asr: unsupported configuration: \(reason)."
        case .unsupportedBatch(let size):
            return "Qwen3Asr: batch size \(size) is not supported. Pass one utterance per call."
        case .invalidInput(let reason):
            return "Qwen3Asr: invalid input: \(reason)."
        case .audioTokenMismatch(let placeholders, let encoderRows):
            return
                "Qwen3Asr: prompt has \(placeholders) audio placeholders but the encoder produced \(encoderRows) rows."
        case .unexpectedAudio:
            return "Qwen3Asr: the prompt has audio placeholders but no audio was given."
        case .unsupportedTask(let name):
            return "Qwen3Asr: \(name) is not supported by this model."
        }
    }
}

/// `audio_config` of Qwen3ASR: a Whisper-style conv stem and a transformer encoder.
public struct SharpAIQwen3AsrAudioConfiguration: Decodable, Sendable {
    public var numMelBins: Int = 128
    public var encoderLayers: Int = 24
    public var encoderAttentionHeads: Int = 16
    public var encoderFFNDim: Int = 4096
    public var dModel: Int = 1024
    public var activationFunction: String = "gelu"
    public var nWindow: Int = 50
    public var nWindowInfer: Int = 800
    public var downsampleHiddenSize: Int = 480
    public var outputDim: Int = 3584
    /// Length of the sinusoid table. Only the first 13 rows are used for a 2 * n_window chunk.
    public var maxSourcePositions: Int = 1500

    public init() {}

    enum CodingKeys: String, CodingKey {
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFFNDim = "encoder_ffn_dim"
        case dModel = "d_model"
        case activationFunction = "activation_function"
        case nWindow = "n_window"
        case nWindowInfer = "n_window_infer"
        case downsampleHiddenSize = "downsample_hidden_size"
        case outputDim = "output_dim"
        case maxSourcePositions = "max_source_positions"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 24
        encoderAttentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 16
        encoderFFNDim = try c.decodeIfPresent(Int.self, forKey: .encoderFFNDim) ?? 4096
        dModel = try c.decodeIfPresent(Int.self, forKey: .dModel) ?? 1024
        activationFunction =
            try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        nWindow = try c.decodeIfPresent(Int.self, forKey: .nWindow) ?? 50
        nWindowInfer = try c.decodeIfPresent(Int.self, forKey: .nWindowInfer) ?? 800
        downsampleHiddenSize = try c.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
        outputDim = try c.decodeIfPresent(Int.self, forKey: .outputDim) ?? 3584
        let sourcePositions = try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions)
        let positions = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
        maxSourcePositions = sourcePositions ?? positions ?? 1500
    }
}

/// `text_config` of Qwen3ASR: a Qwen3 decoder. Positions are plain RoPE. The interleaved
/// mRoPE section in hub configs is a no-op here because audio and text share one position axis.
public struct SharpAIQwen3AsrTextConfiguration: Decodable, Sendable {
    public var hiddenSize: Int = 2048
    public var hiddenLayers: Int = 28
    public var intermediateSize: Int = 6144
    public var attentionHeads: Int = 16
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var rmsNormEps: Float = 1e-6
    public var vocabularySize: Int = 151936
    public var ropeTheta: Float = 1_000_000
    public var maxPositionEmbeddings: Int = 65536
    public var tieWordEmbeddings: Bool?

    public init() {}

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
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        let heads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        hiddenSize = hidden
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 28
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        attentionHeads = heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (heads > 0 ? hidden / heads : 0)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151936
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 65536
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
    }
}

/// Top-level Qwen3ASR configuration. Covers both the ASR model and the forced aligner.
/// The forced aligner is selected when `thinker_config.model_type` is
/// `qwen3_forced_aligner` or when `classify_num` is set.
public struct SharpAIQwen3AsrConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public var audioConfiguration: SharpAIQwen3AsrAudioConfiguration
    public var textConfiguration: SharpAIQwen3AsrTextConfiguration
    public var audioTokenId: Int = 151676
    public var audioStartTokenId: Int = 151669
    public var audioEndTokenId: Int = 151670
    public var timestampTokenId: Int = 151705
    /// Milliseconds per aligner timestamp class.
    public var timestampSegmentTime: Double = 80
    /// Size of the forced aligner's classification head. Unused by the ASR model.
    public var classifyNum: Int = 5000
    public var forcedAligner: Bool = false
    public var tieWordEmbeddings: Bool = true

    public init() {
        self.audioConfiguration = SharpAIQwen3AsrAudioConfiguration()
        self.textConfiguration = SharpAIQwen3AsrTextConfiguration()
    }

    enum CodingKeys: String, CodingKey {
        case thinkerConfig = "thinker_config"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case modelType = "model_type"
        case audioTokenId = "audio_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case audioEndTokenId = "audio_end_token_id"
        case timestampTokenId = "timestamp_token_id"
        case timestampSegmentTime = "timestamp_segment_time"
        case classifyNum = "classify_num"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        let source: KeyedDecodingContainer<CodingKeys>
        if root.contains(.thinkerConfig) {
            source = try root.nestedContainer(keyedBy: CodingKeys.self, forKey: .thinkerConfig)
        } else {
            source = root
        }

        let thinkerType = try source.decodeIfPresent(String.self, forKey: .modelType)
        let classify = try source.decodeIfPresent(Int.self, forKey: .classifyNum)
        forcedAligner = thinkerType == "qwen3_forced_aligner" || classify != nil
        classifyNum = classify ?? 5000

        let audio =
            try source.decodeIfPresent(SharpAIQwen3AsrAudioConfiguration.self, forKey: .audioConfig)
            ?? SharpAIQwen3AsrAudioConfiguration()
        let text =
            try source.decodeIfPresent(SharpAIQwen3AsrTextConfiguration.self, forKey: .textConfig)
            ?? SharpAIQwen3AsrTextConfiguration()
        audioConfiguration = audio
        textConfiguration = text

        audioTokenId = try source.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151676
        audioStartTokenId =
            try source.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 151669
        audioEndTokenId = try source.decodeIfPresent(Int.self, forKey: .audioEndTokenId) ?? 151670
        timestampTokenId =
            try source.decodeIfPresent(Int.self, forKey: .timestampTokenId)
            ?? root.decodeIfPresent(Int.self, forKey: .timestampTokenId)
            ?? 151705
        timestampSegmentTime =
            try root.decodeIfPresent(Double.self, forKey: .timestampSegmentTime)
            ?? source.decodeIfPresent(Double.self, forKey: .timestampSegmentTime)
            ?? 80

        tieWordEmbeddings =
            try source.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? root.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? text.tieWordEmbeddings
            ?? true
    }

    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ reason: String) throws {
            if !condition {
                throw SharpAIQwen3AsrError.unsupportedConfiguration(reason)
            }
        }

        let audio = audioConfiguration
        let text = textConfiguration
        let chunk = audio.nWindow * 2

        try require(audio.activationFunction == "gelu", "audio activation must be gelu")
        try require(audio.numMelBins >= 8, "num_mel_bins must be at least 8")
        try require(audio.dModel > 0, "d_model must be positive")
        try require(audio.encoderLayers > 0, "encoder_layers must be positive")
        try require(audio.encoderAttentionHeads > 0, "encoder_attention_heads must be positive")
        try require(
            audio.dModel % audio.encoderAttentionHeads == 0,
            "d_model must be divisible by encoder_attention_heads")
        try require(audio.nWindow > 0, "n_window must be positive")
        try require(
            audio.nWindowInfer >= chunk && audio.nWindowInfer % chunk == 0,
            "n_window_infer must be a positive multiple of 2 * n_window")
        try require(audio.downsampleHiddenSize > 0, "downsample_hidden_size must be positive")
        try require(audio.maxSourcePositions >= 13, "max_source_positions must be at least 13")
        try require(
            audio.outputDim == text.hiddenSize,
            "audio output_dim must equal the text hidden_size")

        try require(text.hiddenSize > 0, "hidden_size must be positive")
        try require(text.hiddenLayers > 0, "num_hidden_layers must be positive")
        try require(text.attentionHeads > 0, "num_attention_heads must be positive")
        try require(text.kvHeads > 0, "num_key_value_heads must be positive")
        try require(
            text.attentionHeads % text.kvHeads == 0,
            "num_attention_heads must be divisible by num_key_value_heads")
        try require(text.headDim > 0, "head_dim must be positive")
        try require(text.vocabularySize > 0, "vocab_size must be positive")
        try require(classifyNum > 0, "classify_num must be positive")
    }
}
