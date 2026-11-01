// SharpAI fork overlay: ModernVBERT configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

/// Config for `model_type` "modernvbert" (`ModernVBertModel`, `ColModernVBertForRetrieval`).
///
/// The text tower is a ModernBERT encoder (bidirectional). The vision tower is SigLIP.
/// A pixel-shuffle connector joins them. UNVERIFIED: top-level key names `image_token_id`,
/// `pixel_shuffle_factor` and `embedding_dim`, and the defaults, are not checked against a
/// real ModernVBERT config file.
public struct SharpAIModernvbertConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    /// ModernBERT encoder. Attention layers repeat every `global_attn_every_n_layers`.
    public struct TextConfiguration: Decodable, Sendable {
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _normEps: Float?
        public var normEps: Float { _normEps ?? 1e-5 }
        private let _normBias: Bool?
        public var normBias: Bool { _normBias ?? false }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? false }
        private let _mlpBias: Bool?
        public var mlpBias: Bool { _mlpBias ?? false }
        private let _globalAttnEveryNLayers: Int?
        public var globalAttnEveryNLayers: Int { _globalAttnEveryNLayers ?? 3 }
        private let _localAttention: Int?
        public var localAttention: Int { _localAttention ?? 128 }
        private let _globalRopeTheta: Float?
        private let _ropeTheta: Float?
        /// UNVERIFIED: default 160000 is ModernBERT-base's global base. Falls back to `rope_theta`.
        public var globalRopeTheta: Float { _globalRopeTheta ?? _ropeTheta ?? 160_000 }
        private let _localRopeTheta: Float?
        /// UNVERIFIED: default 10000 is ModernBERT-base's local base.
        public var localRopeTheta: Float { _localRopeTheta ?? 10_000 }

        public var headDim: Int { hiddenSize / attentionHeads }

        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _normEps = "norm_eps"
            case _normBias = "norm_bias"
            case _attentionBias = "attention_bias"
            case _mlpBias = "mlp_bias"
            case _globalAttnEveryNLayers = "global_attn_every_n_layers"
            case _localAttention = "local_attention"
            case _globalRopeTheta = "global_rope_theta"
            case _ropeTheta = "rope_theta"
            case _localRopeTheta = "local_rope_theta"
        }
    }

    /// SigLIP vision tower. Output width is `hidden_size`, before the connector.
    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let imageSize: Int
        private let _channels: Int?
        public var channels: Int { _channels ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    /// Token id that marks an image slot in the prompt.
    public let imageTokenIndex: Int
    /// Vision patches merged per side into one image token. UNVERIFIED default 4.
    public let pixelShuffleFactor: Int
    /// Width of each output token vector for `ColModernVBertForRetrieval`. Default 128.
    public let embeddingDimensions: Int
    /// False for plain `ModernVBertModel` checkpoints, which have no retrieval projection.
    public let hasRetrievalHead: Bool

    /// Image tokens produced for one image: `(image_size / patch_size / factor)^2`.
    public var imageTokenCount: Int {
        let side = visionConfiguration.imageSize / visionConfiguration.patchSize / pixelShuffleFactor
        return side * side
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_id"
        case pixelShuffleFactor = "pixel_shuffle_factor"
        case embeddingDimensions = "embedding_dim"
        case architectures
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "modernvbert"
        textConfiguration = try container.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try container.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenIndex = try container.decode(Int.self, forKey: .imageTokenIndex)
        pixelShuffleFactor =
            try container.decodeIfPresent(Int.self, forKey: .pixelShuffleFactor) ?? 4
        embeddingDimensions =
            try container.decodeIfPresent(Int.self, forKey: .embeddingDimensions) ?? 128
        let architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hasRetrievalHead =
            architectures.isEmpty || architectures.contains("ColModernVBertForRetrieval")
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let vision = visionConfiguration

        guard text.vocabularySize > 0, text.hiddenSize > 0, text.intermediateSize > 0,
            text.hiddenLayers > 0, text.attentionHeads > 0, text.localAttention > 0,
            text.globalAttnEveryNLayers > 0
        else {
            throw SharpAIModernvbertError.invalidConfiguration("text_config sizes must be positive")
        }
        guard text.hiddenSize % text.attentionHeads == 0 else {
            throw SharpAIModernvbertError.invalidConfiguration(
                "text hidden_size must be divisible by num_attention_heads")
        }
        guard vision.hiddenSize > 0, vision.hiddenLayers > 0, vision.intermediateSize > 0,
            vision.attentionHeads > 0, vision.patchSize > 0, vision.imageSize > 0
        else {
            throw SharpAIModernvbertError.invalidConfiguration("vision_config sizes must be positive")
        }
        guard vision.hiddenSize % vision.attentionHeads == 0 else {
            throw SharpAIModernvbertError.invalidConfiguration(
                "vision hidden_size must be divisible by num_attention_heads")
        }
        guard vision.imageSize % vision.patchSize == 0 else {
            throw SharpAIModernvbertError.invalidConfiguration(
                "vision image_size must be divisible by patch_size")
        }
        let grid = vision.imageSize / vision.patchSize
        guard pixelShuffleFactor > 0, grid % pixelShuffleFactor == 0 else {
            throw SharpAIModernvbertError.invalidConfiguration(
                "patch grid (\(grid)) must be divisible by pixel_shuffle_factor (\(pixelShuffleFactor))")
        }
        guard embeddingDimensions > 0 else {
            throw SharpAIModernvbertError.invalidConfiguration(
                "embedding_dim must be positive, got \(embeddingDimensions)")
        }
    }
}

public enum SharpAIModernvbertError: Error, Equatable {
    case invalidConfiguration(String)
    case imageTokenMismatch(expected: Int, actual: Int)
}
