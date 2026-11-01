// SharpAI fork overlay: cohere_asr configuration. Not part of upstream mlx-swift-lm.
//
// Defaults follow transformers' CohereAsrConfig and its ParakeetEncoder defaults
// (CohereAsrConfig._default_encoder_config_kwargs). The Hub config is gated, so
// missing keys fall back to these values.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum SharpAICohereAsrError: LocalizedError {
    case unsupportedActivation(String)
    case unsupportedConfiguration(String)
    case missingAudioInput
    case missingEncoderStates
    case missingWeight(String)
    case sequenceTooLong(length: Int, limit: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedActivation(let name):
            return "CohereAsr: unsupported activation '\(name)'."
        case .unsupportedConfiguration(let reason):
            return "CohereAsr: unsupported configuration: \(reason)."
        case .missingAudioInput:
            return "CohereAsr: prepare needs audio features in LMInput.audio."
        case .missingEncoderStates:
            return "CohereAsr: decoding needs encoder states from prepare(_:cache:state:prefill:)."
        case .missingWeight(let key):
            return "CohereAsr: checkpoint is missing '\(key)'."
        case .sequenceTooLong(let length, let limit):
            return "CohereAsr: sequence length \(length) exceeds the limit \(limit)."
        }
    }
}

/// Resolves the activation names used by transformers' `ACT2FN` that this model needs.
enum SharpAICohereAsrActivation {
    static func resolve(_ name: String) throws -> (MLXArray) -> MLXArray {
        switch name {
        case "relu":
            return { relu($0) }
        case "silu", "swish":
            return { silu($0) }
        case "gelu":
            return { gelu($0) }
        default:
            throw SharpAICohereAsrError.unsupportedActivation(name)
        }
    }
}

/// `encoder_config` of `CohereAsrConfig`: a ParakeetEncoder (FastConformer).
public struct SharpAICohereAsrEncoderConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 1280
    public var hiddenLayers: Int = 48
    public var attentionHeads: Int = 8
    public var intermediateSize: Int = 5120
    public var hiddenAct: String = "silu"
    public var attentionBias: Bool = true
    public var convolutionBias: Bool = true
    public var convKernelSize: Int = 9
    public var subsamplingFactor: Int = 8
    public var subsamplingConvChannels: Int = 256
    public var numMelBins: Int = 128
    public var subsamplingConvKernelSize: Int = 3
    public var subsamplingConvStride: Int = 2
    public var maxPositionEmbeddings: Int = 5000
    public var scaleInput: Bool = false

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case convolutionBias = "convolution_bias"
        case convKernelSize = "conv_kernel_size"
        case subsamplingFactor = "subsampling_factor"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case numMelBins = "num_mel_bins"
        case subsamplingConvKernelSize = "subsampling_conv_kernel_size"
        case subsamplingConvStride = "subsampling_conv_stride"
        case maxPositionEmbeddings = "max_position_embeddings"
        case scaleInput = "scale_input"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        convolutionBias = try c.decodeIfPresent(Bool.self, forKey: .convolutionBias) ?? true
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 9
        subsamplingFactor = try c.decodeIfPresent(Int.self, forKey: .subsamplingFactor) ?? 8
        subsamplingConvChannels =
            try c.decodeIfPresent(Int.self, forKey: .subsamplingConvChannels) ?? 256
        numMelBins = try c.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        subsamplingConvKernelSize =
            try c.decodeIfPresent(Int.self, forKey: .subsamplingConvKernelSize) ?? 3
        subsamplingConvStride =
            try c.decodeIfPresent(Int.self, forKey: .subsamplingConvStride) ?? 2
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 5000
        scaleInput = try c.decodeIfPresent(Bool.self, forKey: .scaleInput) ?? false
    }
}

/// `CohereAsrConfig`: a FastConformer encoder and a causal transformer decoder with
/// cross-attention. Decoder self-attention uses `num_key_value_heads` equal to the
/// attention head count only (see `validateModelConfiguration`).
public struct SharpAICohereAsrConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public var modelType: String = "cohere_asr"
    public var encoderConfiguration: SharpAICohereAsrEncoderConfiguration
    public var vocabularySize: Int = 16384
    public var hiddenSize: Int = 1024
    public var hiddenLayers: Int = 8
    public var attentionHeads: Int = 8
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 4096
    public var hiddenAct: String = "relu"
    public var maxPositionEmbeddings: Int = 1024
    public var padTokenId: Int? = 2
    public var eosTokenId: Int? = 3
    public var bosTokenId: Int? = 4
    public var decoderStartTokenId: Int?
    public var tieWordEmbeddings: Bool = false
    public var attentionBias: Bool = true

    public init() {
        self.encoderConfiguration = SharpAICohereAsrEncoderConfiguration()
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoderConfiguration = "encoder_config"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenId = "pad_token_id"
        case eosTokenId = "eos_token_id"
        case bosTokenId = "bos_token_id"
        case decoderStartTokenId = "decoder_start_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere_asr"
        encoderConfiguration =
            try c.decodeIfPresent(
                SharpAICohereAsrEncoderConfiguration.self, forKey: .encoderConfiguration)
            ?? SharpAICohereAsrEncoderConfiguration()
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 16384
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 8
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (attentionHeads > 0 ? hiddenSize / attentionHeads : 0)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "relu"
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1024
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 2
        eosTokenId = try c.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 3
        bosTokenId = try c.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 4
        decoderStartTokenId = try c.decodeIfPresent(Int.self, forKey: .decoderStartTokenId)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
    }

    public func validateModelConfiguration() throws {
        func require(_ condition: Bool, _ reason: String) throws {
            if !condition {
                throw SharpAICohereAsrError.unsupportedConfiguration(reason)
            }
        }

        let enc = encoderConfiguration
        try require(vocabularySize > 0, "vocab_size must be positive")
        try require(hiddenLayers > 0, "num_hidden_layers must be positive")
        try require(attentionHeads > 0, "num_attention_heads must be positive")
        try require(
            kvHeads == attentionHeads,
            "grouped-query decoder attention (num_key_value_heads != num_attention_heads) is not implemented")
        try require(headDim > 0, "head_dim must be positive")
        _ = try SharpAICohereAsrActivation.resolve(hiddenAct)

        try require(enc.hiddenLayers > 0, "encoder num_hidden_layers must be positive")
        try require(enc.attentionHeads > 0, "encoder num_attention_heads must be positive")
        try require(
            enc.hiddenSize % enc.attentionHeads == 0,
            "encoder hidden_size must be divisible by num_attention_heads")
        _ = try SharpAICohereAsrActivation.resolve(enc.hiddenAct)
        try require(enc.convKernelSize % 2 == 1, "encoder conv_kernel_size must be odd")
        try require(
            enc.subsamplingConvKernelSize % 2 == 1,
            "encoder subsampling_conv_kernel_size must be odd")
        try require(enc.subsamplingConvStride >= 1, "subsampling_conv_stride must be positive")
        try require(
            enc.subsamplingFactor >= 2 && (enc.subsamplingFactor & (enc.subsamplingFactor - 1)) == 0,
            "subsampling_factor must be a power of two >= 2")
        let subsamplingLayers = Int(log2(Double(enc.subsamplingFactor)))
        let melAfterSubsampling =
            enc.numMelBins / Int(pow(Double(enc.subsamplingConvStride), Double(subsamplingLayers)))
        try require(melAfterSubsampling > 0, "num_mel_bins too small for subsampling_factor")
    }
}
