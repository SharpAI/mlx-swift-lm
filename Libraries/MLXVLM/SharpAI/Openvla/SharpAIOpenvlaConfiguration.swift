// SharpAI fork overlay: OpenVLA configuration. Not part of upstream mlx-swift-lm.

import Foundation
import MLXLMCommon

/// Config for `model_type` "openvla" (OpenVLA-7B: Llama-2 text, fused DINOv2 + SigLIP vision).
///
/// Only the `dinosiglip-vit-so-224px` backbone and the `no-align+fused-gelu-mlp` projector are
/// implemented. Other values are rejected by `validateModelConfiguration()`.
public struct SharpAIOpenvlaConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public static let supportedVisionBackboneId = "dinosiglip-vit-so-224px"
    public static let supportedArchSpecifier = "no-align+fused-gelu-mlp"
    public static let supportedImageSizes = [224, 224]

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        public let headDim: Int?
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool

        public var resolvedHeadDim: Int { headDim ?? hiddenSize / attentionHeads }

        /// Defaults for any field the checkpoint's `text_config` omits. Llama-2-7B shapes, and
        /// HF `LlamaConfig` for `rms_norm_eps`. UNVERIFIED: that eps default is recalled, not read.
        public init(
            hiddenSize: Int = 4096,
            hiddenLayers: Int = 32,
            intermediateSize: Int = 11008,
            attentionHeads: Int = 32,
            kvHeads: Int? = nil,
            vocabularySize: Int = 32064,
            headDim: Int? = nil,
            rmsNormEps: Float = 1e-6,
            ropeTheta: Float = 10_000,
            ropeTraditional: Bool = false
        ) {
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads ?? attentionHeads
            self.vocabularySize = vocabularySize
            self.headDim = headDim
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.ropeTraditional = ropeTraditional
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
        }

        public init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let defaults = TextConfiguration()
            let heads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? defaults.attentionHeads
            self.init(
                hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                    ?? defaults.hiddenSize,
                hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                    ?? defaults.hiddenLayers,
                intermediateSize: try container.decodeIfPresent(
                    Int.self, forKey: .intermediateSize) ?? defaults.intermediateSize,
                attentionHeads: heads,
                kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads,
                vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                    ?? defaults.vocabularySize,
                headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
                rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                    ?? defaults.rmsNormEps,
                ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                    ?? defaults.ropeTheta,
                ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                    ?? defaults.ropeTraditional
            )
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let padTokenId: Int
    public let visionBackboneId: String
    public let archSpecifier: String
    public let imageSizes: [Int]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case padTokenId = "pad_token_id"
        case visionBackboneId = "vision_backbone_id"
        case archSpecifier = "arch_specifier"
        case imageSizes = "image_sizes"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "openvla"
        textConfiguration =
            try container.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()
        padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 32000
        visionBackboneId =
            try container.decodeIfPresent(String.self, forKey: .visionBackboneId)
            ?? Self.supportedVisionBackboneId
        archSpecifier =
            try container.decodeIfPresent(String.self, forKey: .archSpecifier)
            ?? Self.supportedArchSpecifier
        imageSizes =
            try container.decodeIfPresent([Int].self, forKey: .imageSizes)
            ?? Self.supportedImageSizes
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.intermediateSize > 0,
            text.attentionHeads > 0, text.kvHeads > 0, text.vocabularySize > 0,
            text.resolvedHeadDim > 0
        else {
            throw SharpAIOpenvlaError.invalidConfiguration("text_config sizes must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIOpenvlaError.invalidConfiguration(
                "num_attention_heads must be a multiple of num_key_value_heads")
        }
        guard visionBackboneId == Self.supportedVisionBackboneId else {
            throw SharpAIOpenvlaError.invalidConfiguration(
                "unsupported vision_backbone_id '\(visionBackboneId)'")
        }
        guard archSpecifier == Self.supportedArchSpecifier else {
            throw SharpAIOpenvlaError.invalidConfiguration(
                "unsupported arch_specifier '\(archSpecifier)'")
        }
        guard imageSizes == Self.supportedImageSizes else {
            throw SharpAIOpenvlaError.invalidConfiguration(
                "image_sizes must be [224, 224], got \(imageSizes)")
        }
    }
}

public enum SharpAIOpenvlaError: Error, Equatable {
    case invalidConfiguration(String)
    case imageChannelMismatch(Int)
    case imageSizeMismatch(expected: Int, height: Int, width: Int)
    case visionPositionMismatch(expected: Int, actual: Int)
    case batchMismatch(text: Int, image: Int)
}
