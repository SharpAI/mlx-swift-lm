// SharpAI fork overlay: qwen3_omni_moe configuration. Not part of upstream mlx-swift-lm.
// Reads the thinker block (`thinker_config`, or the same keys at top level). Talker and
// code2wav blocks are ignored. UNVERIFIED: field names and defaults follow transformers'
// Qwen3OmniMoe config classes as recalled, not a checkpoint on disk.

import Foundation
import MLXLMCommon

public enum SharpAIQwen3OmniMoeError: LocalizedError {
    case unsupported(String)
    case invalidInput(String)
    case featureTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "Qwen3OmniMoe unsupported configuration: \(detail)"
        case .invalidInput(let detail):
            return "Qwen3OmniMoe invalid input: \(detail)"
        case .featureTokenMismatch(let placeholders, let features):
            return "Qwen3OmniMoe has \(placeholders) placeholder tokens, \(features) embeddings."
        }
    }
}

/// `thinker_config.audio_config` (`model_type: qwen3_omni_moe_audio_encoder`).
public struct SharpAIQwen3OmniMoeAudioConfiguration: Decodable, Sendable {
    public let dModel: Int
    public let encoderLayers: Int
    public let encoderAttentionHeads: Int
    public let encoderFfnDim: Int
    public let numMelBins: Int
    public let maxSourcePositions: Int
    /// Half the chunk length in mel frames: each audio chunk spans `2 * nWindow` frames.
    public let nWindow: Int
    /// Attention window length in mel frames. A multiple of `2 * nWindow`.
    public let nWindowInfer: Int
    public let downsampleHiddenSize: Int
    public let outputDim: Int
    public let activationFunction: String
    public let scaleEmbedding: Bool

    enum CodingKeys: String, CodingKey {
        case dModel = "d_model"
        case encoderLayers = "encoder_layers"
        case encoderAttentionHeads = "encoder_attention_heads"
        case encoderFfnDim = "encoder_ffn_dim"
        case numMelBins = "num_mel_bins"
        case maxSourcePositions = "max_source_positions"
        case nWindow = "n_window"
        case nWindowInfer = "n_window_infer"
        case downsampleHiddenSize = "downsample_hidden_size"
        case outputDim = "output_dim"
        case activationFunction = "activation_function"
        case scaleEmbedding = "scale_embedding"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        dModel = try container.decodeIfPresent(Int.self, forKey: .dModel) ?? 1280
        encoderLayers = try container.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
        encoderAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
        encoderFfnDim = try container.decodeIfPresent(Int.self, forKey: .encoderFfnDim) ?? 5120
        numMelBins = try container.decodeIfPresent(Int.self, forKey: .numMelBins) ?? 128
        maxSourcePositions =
            try container.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        nWindow = try container.decodeIfPresent(Int.self, forKey: .nWindow) ?? 50
        nWindowInfer = try container.decodeIfPresent(Int.self, forKey: .nWindowInfer) ?? 800
        downsampleHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
        outputDim = try container.decodeIfPresent(Int.self, forKey: .outputDim) ?? 2048
        activationFunction =
            try container.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
        scaleEmbedding =
            try container.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
    }
}

/// Thinker blocks read from `thinker_config`, or from the top level when no wrapper is present.
private struct SharpAIQwen3OmniMoeThinkerFields: Decodable {
    let audioConfiguration: SharpAIQwen3OmniMoeAudioConfiguration
    let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
    let textConfiguration: Qwen3VLMoEConfiguration.TextConfiguration
    let audioTokenId: Int
    let imageTokenId: Int
    let videoTokenId: Int
    let visionStartTokenId: Int

    enum CodingKeys: String, CodingKey {
        case audioConfiguration = "audio_config"
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case audioTokenId = "audio_token_id"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        audioConfiguration = try container.decode(
            SharpAIQwen3OmniMoeAudioConfiguration.self, forKey: .audioConfiguration)
        visionConfiguration = try container.decode(
            Qwen3VLConfiguration.VisionConfiguration.self, forKey: .visionConfiguration)
        textConfiguration = try container.decode(
            Qwen3VLMoEConfiguration.TextConfiguration.self, forKey: .textConfiguration)
        audioTokenId = try container.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 151_646
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 151_655
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 151_656
        visionStartTokenId =
            try container.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 151_652
    }
}

/// Top-level `config.json` for `model_type: qwen3_omni_moe`, thinker (text + vision + audio) only.
public struct SharpAIQwen3OmniMoeConfiguration: Decodable, Sendable {
    public let modelType: String
    public let audioConfiguration: SharpAIQwen3OmniMoeAudioConfiguration
    public let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
    public let textConfiguration: Qwen3VLMoEConfiguration.TextConfiguration
    public let audioTokenId: Int
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let visionStartTokenId: Int

    enum RootKeys: String, CodingKey {
        case modelType = "model_type"
        case thinkerConfiguration = "thinker_config"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        let fields: SharpAIQwen3OmniMoeThinkerFields
        if root.contains(.thinkerConfiguration) {
            fields = try root.decode(
                SharpAIQwen3OmniMoeThinkerFields.self, forKey: .thinkerConfiguration)
        } else {
            fields = try SharpAIQwen3OmniMoeThinkerFields(from: decoder)
        }
        modelType =
            try root.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_omni_moe"
        audioConfiguration = fields.audioConfiguration
        visionConfiguration = fields.visionConfiguration
        textConfiguration = fields.textConfiguration
        audioTokenId = fields.audioTokenId
        imageTokenId = fields.imageTokenId
        videoTokenId = fields.videoTokenId
        visionStartTokenId = fields.visionStartTokenId
    }

    /// Builds the Qwen3-VL-MoE configuration that the text decoder is reused from.
    /// Only the text block, vision block and image/video/vision-start token ids are copied.
    public func languageConfiguration() throws -> Qwen3VLMoEConfiguration {
        let bridge = SharpAIQwen3OmniMoeLanguageBridge(
            modelType: "qwen3_vl_moe",
            textConfiguration: textConfiguration,
            visionConfiguration: visionConfiguration,
            imageTokenId: imageTokenId,
            videoTokenId: videoTokenId,
            visionStartTokenId: visionStartTokenId)
        let data = try JSONEncoder().encode(bridge)
        return try JSONDecoder().decode(Qwen3VLMoEConfiguration.self, from: data)
    }
}

/// Encodes the parts of the thinker config that `Qwen3VLMoEConfiguration` decodes.
private struct SharpAIQwen3OmniMoeLanguageBridge: Encodable {
    let modelType: String
    let textConfiguration: Qwen3VLMoEConfiguration.TextConfiguration
    let visionConfiguration: Qwen3VLConfiguration.VisionConfiguration
    let imageTokenId: Int
    let videoTokenId: Int
    let visionStartTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
    }
}

extension SharpAIQwen3OmniMoeConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let audio = audioConfiguration
        let text = textConfiguration
        let vision = visionConfiguration

        guard audio.activationFunction == "gelu" else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "audio activation_function \(audio.activationFunction)")
        }
        guard !audio.scaleEmbedding else {
            throw SharpAIQwen3OmniMoeError.unsupported("audio scale_embedding = true")
        }
        guard audio.dModel > 0, audio.encoderLayers > 0, audio.numMelBins > 0,
            audio.maxSourcePositions > 0, audio.downsampleHiddenSize > 0
        else {
            throw SharpAIQwen3OmniMoeError.unsupported("audio encoder dimensions must be positive")
        }
        guard audio.encoderAttentionHeads > 0, audio.dModel % audio.encoderAttentionHeads == 0
        else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "audio d_model must be divisible by encoder_attention_heads")
        }
        guard audio.nWindow > 0, audio.nWindowInfer >= 2 * audio.nWindow,
            audio.nWindowInfer % (2 * audio.nWindow) == 0
        else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "audio n_window_infer must be a positive multiple of 2 * n_window")
        }
        guard audio.outputDim == text.hiddenSize else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "audio output_dim \(audio.outputDim) must equal text hidden_size \(text.hiddenSize)")
        }
        guard vision.outHiddenSize == text.hiddenSize else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "vision out_hidden_size \(vision.outHiddenSize) must equal text hidden_size \(text.hiddenSize)"
            )
        }
        guard vision.numHeads > 0, vision.hiddenSize % vision.numHeads == 0 else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "vision hidden_size must be divisible by num_heads")
        }
        guard text.numAttentionHeads > 0, text.numKeyValueHeads > 0,
            text.numAttentionHeads % text.numKeyValueHeads == 0
        else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "text num_attention_heads must be divisible by num_key_value_heads")
        }
        guard text.numExperts > 0, text.numExpertsPerTok > 0,
            text.numExpertsPerTok <= text.numExperts
        else {
            throw SharpAIQwen3OmniMoeError.unsupported(
                "text num_experts_per_tok must be between 1 and num_experts")
        }
    }
}
