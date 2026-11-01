// SharpAI fork overlay: vibevoice_asr configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow transformers' VibeVoiceAsrConfig, its acoustic tokenizer encoder config,
// and Qwen2Config for text_config. Missing keys fall back to those defaults.
// Written without compiling. UNVERIFIED: the defaults are copied from transformers 5.3.0.dev0.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum SharpAIVibevoiceAsrError: LocalizedError {
    case unsupportedActivation(String)
    case unsupportedConfiguration(String)
    case missingAudioInput
    case unsupportedAudioShape([Int])
    case audioTokenMismatch(placeholders: Int, frames: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedActivation(let name):
            return "VibevoiceAsr: unsupported activation '\(name)'."
        case .unsupportedConfiguration(let reason):
            return "VibevoiceAsr: unsupported configuration: \(reason)."
        case .missingAudioInput:
            return "VibevoiceAsr: prepare needs audio samples in LMInput.audio."
        case .unsupportedAudioShape(let shape):
            return "VibevoiceAsr: audio must be (samples), (batch, samples) or (batch, 1, samples); got \(shape)."
        case .audioTokenMismatch(let placeholders, let frames):
            return
                "VibevoiceAsr: the prompt has \(placeholders) audio placeholder tokens but the audio encodes \(frames) frames."
        }
    }
}

func sharpAIVibevoiceAsrRequire(_ condition: Bool, _ reason: String) throws {
    if !condition {
        throw SharpAIVibevoiceAsrError.unsupportedConfiguration(reason)
    }
}

/// Resolves the `ACT2FN` names this model uses.
enum SharpAIVibevoiceAsrActivation {
    static func resolve(_ name: String) throws -> (MLXArray) -> MLXArray {
        switch name {
        case "gelu":
            return { gelu($0) }
        case "silu", "swish":
            return { silu($0) }
        case "relu":
            return { relu($0) }
        default:
            throw SharpAIVibevoiceAsrError.unsupportedActivation(name)
        }
    }
}

/// `vibevoice_acoustic_tokenizer_encoder`: a causal conv stem, six strided causal conv stages,
/// and a causal conv head. Each stage is followed by ConvNeXt-style blocks.
public struct SharpAIVibevoiceAsrTokenizerConfiguration: Codable, Sendable {
    public var channels: Int = 1
    public var hiddenSize: Int = 64
    public var kernelSize: Int = 7
    public var rmsNormEps: Float = 1e-5
    public var layerScaleInitValue: Float = 1e-6
    public var numFilters: Int = 32
    public var downsamplingRatios: [Int] = [2, 2, 4, 5, 5, 8]
    public var depths: [Int] = [3, 3, 3, 3, 3, 3, 8]
    public var hiddenAct: String = "gelu"
    public var ffnExpansion: Int = 4
    public var vaeStd: Float = 0.625

    /// Audio samples per latent frame. The processor expands one audio token per hop.
    public var hopLength: Int {
        downsamplingRatios.reduce(1, *)
    }

    /// Default for `semantic_tokenizer_encoder_config`: the same encoder with 128 latent channels.
    public static var semanticDefault: SharpAIVibevoiceAsrTokenizerConfiguration {
        var config = SharpAIVibevoiceAsrTokenizerConfiguration()
        config.hiddenSize = 128
        return config
    }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case channels
        case hiddenSize = "hidden_size"
        case kernelSize = "kernel_size"
        case rmsNormEps = "rms_norm_eps"
        case layerScaleInitValue = "layer_scale_init_value"
        case numFilters = "num_filters"
        case downsamplingRatios = "downsampling_ratios"
        case depths
        case hiddenAct = "hidden_act"
        case ffnExpansion = "ffn_expansion"
        case vaeStd = "vae_std"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 1
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 64
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 7
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        layerScaleInitValue =
            try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue) ?? 1e-6
        numFilters = try c.decodeIfPresent(Int.self, forKey: .numFilters) ?? 32
        downsamplingRatios =
            try c.decodeIfPresent([Int].self, forKey: .downsamplingRatios) ?? [2, 2, 4, 5, 5, 8]
        depths = try c.decodeIfPresent([Int].self, forKey: .depths) ?? [3, 3, 3, 3, 3, 3, 8]
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        ffnExpansion = try c.decodeIfPresent(Int.self, forKey: .ffnExpansion) ?? 4
        vaeStd = try c.decodeIfPresent(Float.self, forKey: .vaeStd) ?? 0.625
    }

    func validate() throws {
        try sharpAIVibevoiceAsrRequire(channels == 1, "tokenizer channels must be 1")
        try sharpAIVibevoiceAsrRequire(hiddenSize > 0, "tokenizer hidden_size must be positive")
        try sharpAIVibevoiceAsrRequire(kernelSize >= 1, "tokenizer kernel_size must be positive")
        try sharpAIVibevoiceAsrRequire(numFilters > 0, "tokenizer num_filters must be positive")
        try sharpAIVibevoiceAsrRequire(ffnExpansion > 0, "tokenizer ffn_expansion must be positive")
        try sharpAIVibevoiceAsrRequire(
            !downsamplingRatios.isEmpty && downsamplingRatios.allSatisfy { $0 > 0 },
            "tokenizer downsampling_ratios must be non-empty and positive")
        try sharpAIVibevoiceAsrRequire(
            depths.count == downsamplingRatios.count + 1,
            "tokenizer depths needs one entry more than downsampling_ratios")
        try sharpAIVibevoiceAsrRequire(
            depths.allSatisfy { $0 >= 0 }, "tokenizer depths must not be negative")
        _ = try SharpAIVibevoiceAsrActivation.resolve(hiddenAct)
    }
}

/// `text_config`: a Qwen2 decoder with grouped-query attention and biased q, k and v projections.
public struct SharpAIVibevoiceAsrTextConfiguration: Codable, Sendable {
    public var hiddenSize: Int = 3584
    public var hiddenLayers: Int = 28
    public var intermediateSize: Int = 18944
    public var attentionHeads: Int = 28
    public var kvHeads: Int = 4
    public var rmsNormEps: Float = 1e-6
    public var vocabularySize: Int = 152064
    public var ropeTheta: Float = 1_000_000
    public var maxPositionEmbeddings: Int = 131072
    public var tieWordEmbeddings: Bool = false

    public var headDim: Int {
        attentionHeads > 0 ? hiddenSize / attentionHeads : 0
    }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    private enum RopeParametersKey: String, CodingKey {
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3584
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 28
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 18944
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 28
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 152064
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Newer configs put rope_theta under rope_parameters. Older ones use the top-level key.
        var theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        if theta == nil,
            let parameters = try decoder.container(keyedBy: RopeParametersKey.self)
                .decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
        {
            theta = parameters["rope_theta"]?.asFloat()
        }
        ropeTheta = theta ?? 1_000_000
    }
}

/// `VibeVoiceAsrConfig`: two acoustic tokenizer encoders (acoustic and semantic), a projector,
/// and a Qwen2 language model. The audio tokens in the prompt are replaced by projected features.
public struct SharpAIVibevoiceAsrConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public var modelType: String = "vibevoice_asr"
    public var acousticTokenizer: SharpAIVibevoiceAsrTokenizerConfiguration
    public var semanticTokenizer: SharpAIVibevoiceAsrTokenizerConfiguration
    public var text: SharpAIVibevoiceAsrTextConfiguration
    public var audioTokenId: Int = 151648
    public var audioBosTokenId: Int = 151646
    public var audioEosTokenId: Int = 151647
    /// Read from the checkpoint for parity. The Swift model encodes the whole input in one pass.
    public var acousticTokenizerChunkSize: Int = 1_440_000

    public init() {
        self.acousticTokenizer = SharpAIVibevoiceAsrTokenizerConfiguration()
        self.semanticTokenizer = SharpAIVibevoiceAsrTokenizerConfiguration.semanticDefault
        self.text = SharpAIVibevoiceAsrTextConfiguration()
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case acousticTokenizer = "acoustic_tokenizer_encoder_config"
        case semanticTokenizer = "semantic_tokenizer_encoder_config"
        case text = "text_config"
        case audioTokenId = "audio_token_id"
        case audioBosTokenId = "audio_bos_token_id"
        case audioEosTokenId = "audio_eos_token_id"
        case acousticTokenizerChunkSize = "acoustic_tokenizer_chunk_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vibevoice_asr"
        acousticTokenizer =
            try c.decodeIfPresent(
                SharpAIVibevoiceAsrTokenizerConfiguration.self, forKey: .acousticTokenizer)
            ?? SharpAIVibevoiceAsrTokenizerConfiguration()
        semanticTokenizer =
            try c.decodeIfPresent(
                SharpAIVibevoiceAsrTokenizerConfiguration.self, forKey: .semanticTokenizer)
            ?? SharpAIVibevoiceAsrTokenizerConfiguration.semanticDefault
        text =
            try c.decodeIfPresent(SharpAIVibevoiceAsrTextConfiguration.self, forKey: .text)
            ?? SharpAIVibevoiceAsrTextConfiguration()
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151648
        audioBosTokenId = try c.decodeIfPresent(Int.self, forKey: .audioBosTokenId) ?? 151646
        audioEosTokenId = try c.decodeIfPresent(Int.self, forKey: .audioEosTokenId) ?? 151647
        acousticTokenizerChunkSize =
            try c.decodeIfPresent(Int.self, forKey: .acousticTokenizerChunkSize) ?? 1_440_000
    }

    public func validateModelConfiguration() throws {
        try acousticTokenizer.validate()
        try semanticTokenizer.validate()

        // Both encoders must produce frames on the same grid, or the features cannot be added.
        try sharpAIVibevoiceAsrRequire(
            acousticTokenizer.downsamplingRatios == semanticTokenizer.downsamplingRatios,
            "acoustic and semantic downsampling_ratios must match")

        try sharpAIVibevoiceAsrRequire(text.vocabularySize > 0, "text vocab_size must be positive")
        try sharpAIVibevoiceAsrRequire(text.hiddenLayers > 0, "text num_hidden_layers must be positive")
        try sharpAIVibevoiceAsrRequire(text.attentionHeads > 0, "text num_attention_heads must be positive")
        try sharpAIVibevoiceAsrRequire(text.kvHeads > 0, "text num_key_value_heads must be positive")
        try sharpAIVibevoiceAsrRequire(
            text.hiddenSize % text.attentionHeads == 0,
            "text hidden_size must be divisible by num_attention_heads")
        try sharpAIVibevoiceAsrRequire(
            text.attentionHeads % text.kvHeads == 0,
            "text num_attention_heads must be a multiple of num_key_value_heads")
        try sharpAIVibevoiceAsrRequire(
            audioTokenId >= 0 && audioTokenId < text.vocabularySize,
            "audio_token_id must index the text vocabulary")
    }
}
