// SharpAI fork overlay: ColPali configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

/// Config for `model_type` "colpali", a retrieval model over a PaliGemma backbone.
///
/// The PaliGemma config sits under `vlm_config`: the vision tower and image token id
/// are read from there. The Gemma text config is top-level `text_config`, with
/// `vlm_config.text_config` as the fallback.
public struct SharpAIColpaliConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        public let headDim: Int?
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool { _ropeTraditional ?? false }

        public var resolvedHeadDim: Int { headDim ?? hiddenSize / attentionHeads }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case headDim = "head_dim"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let imageSize: Int
        /// Output width of the projector. Must equal the text `hidden_size`.
        public let projectionDimensions: Int
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
            case projectionDimensions = "projection_dim"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public struct VLMConfiguration: Decodable, Sendable {
        public let visionConfiguration: VisionConfiguration
        public let textConfiguration: TextConfiguration?
        public let imageTokenIndex: Int
        private let _padTokenId: Int?
        public var padTokenId: Int { _padTokenId ?? 0 }

        enum CodingKeys: String, CodingKey {
            case visionConfiguration = "vision_config"
            case textConfiguration = "text_config"
            case imageTokenIndex = "image_token_index"
            case _padTokenId = "pad_token_id"
        }
    }

    public let modelType: String
    /// Width of each output token vector (`embedding_dim`, 128 for ColPali v1.3).
    public let embeddingDimensions: Int
    public let textConfiguration: TextConfiguration
    public let vlmConfiguration: VLMConfiguration

    public var visionConfiguration: VisionConfiguration { vlmConfiguration.visionConfiguration }
    public var imageTokenIndex: Int { vlmConfiguration.imageTokenIndex }
    public var padTokenId: Int { vlmConfiguration.padTokenId }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case embeddingDimensions = "embedding_dim"
        case textConfiguration = "text_config"
        case vlmConfiguration = "vlm_config"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "colpali"
        embeddingDimensions =
            try container.decodeIfPresent(Int.self, forKey: .embeddingDimensions) ?? 128
        let vlm = try container.decode(VLMConfiguration.self, forKey: .vlmConfiguration)
        vlmConfiguration = vlm

        if let topLevel = try container.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration) {
            textConfiguration = topLevel
        } else if let nested = vlm.textConfiguration {
            textConfiguration = nested
        } else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "colpali config needs text_config or vlm_config.text_config"))
        }
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let vision = visionConfiguration

        guard embeddingDimensions > 0 else {
            throw SharpAIColpaliError.invalidConfiguration(
                "embedding_dim must be positive, got \(embeddingDimensions)")
        }
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.intermediateSize > 0,
            text.attentionHeads > 0, text.kvHeads > 0, text.vocabularySize > 0,
            text.resolvedHeadDim > 0
        else {
            throw SharpAIColpaliError.invalidConfiguration("text_config sizes must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIColpaliError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard vision.hiddenSize > 0, vision.hiddenLayers > 0, vision.intermediateSize > 0,
            vision.attentionHeads > 0, vision.patchSize > 0
        else {
            throw SharpAIColpaliError.invalidConfiguration("vision_config sizes must be positive")
        }
        guard vision.hiddenSize % vision.attentionHeads == 0 else {
            throw SharpAIColpaliError.invalidConfiguration(
                "vision hidden_size must be divisible by num_attention_heads")
        }
        guard vision.imageSize % vision.patchSize == 0 else {
            throw SharpAIColpaliError.invalidConfiguration(
                "vision image_size must be divisible by patch_size")
        }
        guard vision.projectionDimensions == text.hiddenSize else {
            throw SharpAIColpaliError.invalidConfiguration(
                "vision projection_dim (\(vision.projectionDimensions)) must equal text hidden_size (\(text.hiddenSize))")
        }
    }
}

public enum SharpAIColpaliError: Error, Equatable {
    case invalidConfiguration(String)
    case imageTokenMismatch(expected: Int, actual: Int)
}
