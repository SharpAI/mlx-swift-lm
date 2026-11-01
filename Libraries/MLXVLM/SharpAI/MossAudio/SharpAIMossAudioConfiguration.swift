// SharpAI fork overlay: moss_audio configuration. Not part of upstream mlx-swift-lm.
//
// Keys follow OpenMOSS-Team/MOSS-Audio-4B-Instruct config.json (`audio_config`, `language_config`)
// and vLLM's moss_audio.py (`adapter_hidden_size`, `deepstack_num_inject_layers`, `n_window`).
// The text backbone under `language_config` is Qwen3. It is decoded twice: once as
// `Qwen3Configuration` for validation, and once as the overlay's own text config, because MLXLLM
// does not expose Qwen3's fields outside its module.
// UNVERIFIED: `n_window` is absent from config.json (default 200, from vLLM), and the
// `audio_config` keys that config.json does not list are not checked against a checkpoint.

import Foundation
import MLXLLM
import MLXLMCommon

public enum SharpAIMossAudioError: LocalizedError {
    case unsupported(String)
    case invalidConfiguration(String)
    case invalidInput(String)
    case audioTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "MossAudio unsupported: \(detail)"
        case .invalidConfiguration(let detail):
            return "MossAudio invalid configuration: \(detail)"
        case .invalidInput(let detail):
            return "MossAudio invalid input: \(detail)"
        case .audioTokenMismatch(let placeholders, let features):
            return
                "MossAudio audio token mismatch: \(placeholders) placeholders for \(features) audio tokens"
        }
    }
}

/// `audio_config`: the MOSS-Audio encoder. Defaults are the 4B-Instruct values.
public struct SharpAIMossAudioEncoderConfiguration: Codable, Sendable {
    public let dModel: Int
    public let outputDim: Int
    public let numMelBins: Int
    public let encoderLayers: Int
    public let attentionHeads: Int
    public let ffnDim: Int
    public let downsampleHiddenSize: Int
    public let maxSourcePositions: Int
    public let nWindow: Int
    public let deepstackEncoderLayerIndexes: [Int]
    public let layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case outputDim = "output_dim"
        case numMelBins = "num_mel_bins"
        case encoderLayers = "encoder_layers"
        case attentionHeads = "encoder_attention_heads"
        case ffnDim = "encoder_ffn_dim"
        case downsampleHiddenSize = "downsample_hidden_size"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case deepstackEncoderLayerIndexes = "deepstack_encoder_layer_indexes"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        self.dModel = dModel
        self.outputDim = try container.decodeIfPresent(Int.self, forKey: .outputDim) ?? dModel
        self.numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        self.encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        self.attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 20
        self.ffnDim = try container.decodeIfPresent(Int.self, forKey: .ffnDim) ?? 5120
        self.downsampleHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
        self.maxSourcePositions =
            try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        self.nWindow = try container.decodeIfPresent(Int.self, forKey: .nWindow) ?? 200
        self.deepstackEncoderLayerIndexes =
            try container.decodeIfPresent([Int].self, forKey: .deepstackEncoderLayerIndexes)
            ?? [8, 16, 24]
        self.layerNormEps =
            try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
    }
}

/// `language_config` read as the overlay's own Qwen3 text dimensions.
public struct SharpAIMossAudioTextConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)
        self.hiddenSize = hidden
        self.hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        self.attentionHeads = heads
        self.kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? hidden / heads
        self.vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        self.rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Top-level `config.json` for `model_type: moss_audio`.
public struct SharpAIMossAudioConfiguration: Codable, Sendable {
    public let modelType: String
    public let architectures: [String]
    public let languageConfiguration: Qwen3Configuration
    public let textConfiguration: SharpAIMossAudioTextConfiguration
    public let audioConfiguration: SharpAIMossAudioEncoderConfiguration
    public let adapterHiddenSize: Int
    public let deepstackNumInjectLayers: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case languageConfiguration = "language_config"
        case audioConfiguration = "audio_config"
        case adapterHiddenSize = "adapter_hidden_size"
        case deepstackNumInjectLayers = "deepstack_num_inject_layers"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "moss_audio"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        languageConfiguration = try container.decode(
            Qwen3Configuration.self, forKey: .languageConfiguration)
        textConfiguration = try container.decode(
            SharpAIMossAudioTextConfiguration.self, forKey: .languageConfiguration)
        if let audio = try container.decodeIfPresent(
            SharpAIMossAudioEncoderConfiguration.self, forKey: .audioConfiguration)
        {
            audioConfiguration = audio
        } else {
            // Missing `audio_config`: every field takes its default.
            audioConfiguration = try JSONDecoder().decode(
                SharpAIMossAudioEncoderConfiguration.self, from: Data("{}".utf8))
        }
        adapterHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .adapterHiddenSize) ?? 8192
        deepstackNumInjectLayers =
            try container.decodeIfPresent(Int.self, forKey: .deepstackNumInjectLayers)
    }

    /// DeepStack taps merged into the LM: all of `deepstack_encoder_layer_indexes`, capped by
    /// `deepstack_num_inject_layers` when it is set.
    public var deepstackMergerCount: Int {
        let available = audioConfiguration.deepstackEncoderLayerIndexes.count
        guard let limit = deepstackNumInjectLayers else { return available }
        return max(0, min(limit, available))
    }

    /// Checks the model type, the Qwen3 language config, and the audio shapes the encoder needs.
    public func validateModelConfiguration() throws {
        guard modelType == "moss_audio" else {
            throw SharpAIMossAudioError.invalidConfiguration("model_type must be moss_audio")
        }
        try languageConfiguration.validateModelConfiguration()

        let audio = audioConfiguration
        guard audio.dModel >= 4, audio.attentionHeads > 0, audio.dModel % audio.attentionHeads == 0
        else {
            throw SharpAIMossAudioError.invalidConfiguration(
                "audio d_model must be at least 4 and divisible by encoder_attention_heads")
        }
        guard audio.numMelBins > 0, audio.encoderLayers > 0, audio.nWindow > 0 else {
            throw SharpAIMossAudioError.invalidConfiguration(
                "audio num_mel_bins, encoder_layers and n_window must be positive")
        }
        guard
            audio.deepstackEncoderLayerIndexes.allSatisfy({
                $0 >= 0 && $0 < audio.encoderLayers
            })
        else {
            throw SharpAIMossAudioError.invalidConfiguration(
                "deepstack_encoder_layer_indexes must lie within encoder_layers")
        }
        let text = textConfiguration
        guard text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIMossAudioError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
    }
}
