// SharpAI fork overlay: funaudiochat configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

public enum SharpAIFunaudiochatError: LocalizedError {
    case unsupported(String)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "FunAudioChat unsupported: \(detail)"
        }
    }
}

/// Top-level `config.json` for `model_type: funaudiochat`.
/// Verified against: https://huggingface.co/FunAudioLLM/Fun-Audio-Chat-8B/blob/main/config.json
/// (key names and values for `audio_config` and `text_config` below are taken from that file).
/// The nested `crq_transformer_config` and the other audio/text keys are not decoded yet.
public struct SharpAIFunaudiochatConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioTokenIndex: Int?
    public let audioConfig: AudioConfig?
    public let textConfig: TextConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioTokenIndex = "audio_token_index"
        case audioConfig = "audio_config"
        case textConfig = "text_config"
    }

    /// `audio_config` (`model_type: funaudiochat_audio_encoder`).
    public struct AudioConfig: Codable, Sendable {
        public let encoderLayers: Int
        public let dModel: Int
        public let encoderAttentionHeads: Int
        public let encoderFfnDim: Int
        public let numMelBins: Int
        public let outputDim: Int
        public let groupSize: Int
        public let nWindow: Int
        public let maxSourcePositions: Int
        public let codebookSize: Int
        public let enableAudioInvertTower: Bool?
        public let continuousFeaturesMode: String?

        enum CodingKeys: String, CodingKey {
            case encoderLayers = "encoder_layers"
            case dModel = "d_model"
            case encoderAttentionHeads = "encoder_attention_heads"
            case encoderFfnDim = "encoder_ffn_dim"
            case numMelBins = "num_mel_bins"
            case outputDim = "output_dim"
            case groupSize = "group_size"
            case nWindow = "n_window"
            case maxSourcePositions = "max_source_positions"
            case codebookSize = "codebook_size"
            case enableAudioInvertTower = "enable_audio_invert_tower"
            case continuousFeaturesMode = "continuous_features_mode"
        }
    }

    /// `text_config` (`model_type: qwen3`, `architectures: Qwen3ForCausalLM`).
    public struct TextConfig: Codable, Sendable {
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let vocabSize: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Double
        public let ropeTheta: Double
        public let hiddenAct: String
        public let attentionBias: Bool
        public let audioBosIndex: Int?
        public let audioEosIndex: Int?
        public let silIndex: Int?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vocabSize = "vocab_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case hiddenAct = "hidden_act"
            case attentionBias = "attention_bias"
            case audioBosIndex = "audio_bos_index"
            case audioEosIndex = "audio_eos_index"
            case silIndex = "sil_index"
        }
    }

    public func validateModelConfiguration() throws {
        guard modelType == "funaudiochat" else {
            throw SharpAIFunaudiochatError.unsupported("model_type \(modelType)")
        }
    }
}
