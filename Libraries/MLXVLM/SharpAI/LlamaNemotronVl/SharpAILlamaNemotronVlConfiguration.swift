// SharpAI fork overlay: LlamaNemotronVL embedder configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

/// Config for `model_type` "llama_nemotron_vl", the VLM embedder (SigLIP vision, bidirectional Llama).
///
/// UNVERIFIED: the top-level key names (`llm_config` or `text_config`, `img_context_token_id`) are
/// not checked against a real checkpoint. The text config is read from either key.
public struct SharpAILlamaNemotronVlConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _headDim: Int?
        private let _rmsNormEps: Float?
        private let _ropeTheta: Float?
        private let _ropeTraditional: Bool?
        public let ropeScaling: [String: StringOrNumber]?

        public var headDim: Int { _headDim ?? hiddenSize / attentionHeads }
        public var rmsNormEps: Float { _rmsNormEps ?? 1e-5 }
        public var ropeTheta: Float { _ropeTheta ?? 10_000 }
        public var ropeTraditional: Bool { _ropeTraditional ?? false }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _headDim = "head_dim"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
            intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
            kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
            vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
            _headDim = try container.decodeIfPresent(Int.self, forKey: ._headDim)
            _rmsNormEps = try container.decodeIfPresent(Float.self, forKey: ._rmsNormEps)
            _ropeTheta = try container.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _ropeTraditional = try container.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
            ropeScaling = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let imageSize: Int
        private let _channels: Int?
        private let _layerNormEps: Float?

        public var channels: Int { _channels ?? 3 }
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
    /// Pixel-shuffle factor. Must be `1 / k` for a whole number `k` (0.5 gives k = 2).
    public let downsampleRatio: Float
    /// Pixel-shuffle layout. "v1" skips the second transpose, any other value uses it.
    public let psVersion: String
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case llmConfiguration = "llm_config"
        case visionConfiguration = "vision_config"
        case downsampleRatio = "downsample_ratio"
        case psVersion = "ps_version"
        case imageTokenId = "img_context_token_id"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "llama_nemotron_vl"
        if let llm = try container.decodeIfPresent(TextConfiguration.self, forKey: .llmConfiguration) {
            textConfiguration = llm
        } else {
            textConfiguration = try container.decode(TextConfiguration.self, forKey: .textConfiguration)
        }
        visionConfiguration = try container.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        downsampleRatio = try container.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        psVersion = try container.decodeIfPresent(String.self, forKey: .psVersion) ?? "v2"
        imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
    }

    /// Integer `k` with `downsample_ratio == 1 / k`. Returns 0 when no valid `k` exists, so validation can throw.
    public var pixelShuffleFactor: Int {
        guard downsampleRatio.isFinite, downsampleRatio > 0 else { return 0 }
        let k = (1 / downsampleRatio).rounded()
        guard k < 1_000_000 else { return 0 }
        return Int(k)
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let vision = visionConfiguration

        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.intermediateSize > 0,
            text.attentionHeads > 0, text.kvHeads > 0, text.vocabularySize > 0, text.headDim > 0
        else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration("text config sizes must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard vision.hiddenSize > 0, vision.hiddenLayers > 0, vision.intermediateSize > 0,
            vision.attentionHeads > 0, vision.patchSize > 0, vision.imageSize > 0
        else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration("vision config sizes must be positive")
        }
        guard vision.hiddenSize % vision.attentionHeads == 0 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration(
                "vision hidden_size must be divisible by num_attention_heads")
        }
        guard vision.imageSize % vision.patchSize == 0 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration(
                "vision image_size must be divisible by patch_size")
        }
        let factor = pixelShuffleFactor
        guard factor >= 1, abs(1 / Float(factor) - downsampleRatio) < 1e-6 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration(
                "downsample_ratio must be 1/k for a whole number k, got \(downsampleRatio)")
        }
        let grid = vision.imageSize / vision.patchSize
        guard grid % factor == 0 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration(
                "vision grid (\(grid)) must be divisible by the pixel-shuffle factor (\(factor))")
        }
        guard imageTokenId >= 0 else {
            throw SharpAILlamaNemotronVlError.invalidConfiguration("img_context_token_id must be >= 0")
        }
    }
}

public enum SharpAILlamaNemotronVlError: Error, Equatable {
    case invalidConfiguration(String)
    case imageTokenMismatch(expected: Int, actual: Int)
}
