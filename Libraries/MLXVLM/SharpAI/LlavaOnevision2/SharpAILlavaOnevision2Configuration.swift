// SharpAI fork overlay: llava_onevision2 configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the top-level config and the nested `text_config` and `vision_config` blocks.
// Field names follow lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct config.json and
// configuration_llava_onevision2.py (checked against the repo's modeling source).

import Foundation
import MLXLMCommon

public enum SharpAILlavaOnevision2Error: LocalizedError {
    case duplicateWeightKey(String)
    case imageTokenMismatch(placeholders: Int, features: Int)
    case patchCountMismatch(patches: Int, grid: Int)
    case missingImageGrid
    case videoNotImplemented
    case emptyPrompt
    case batchSizeNotSupported(Int)

    public var errorDescription: String? {
        switch self {
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after llava_onevision2 sanitize: \(key)"
        case .imageTokenMismatch(let placeholders, let features):
            return
                "llava_onevision2: \(placeholders) image placeholder tokens but \(features) merged image features"
        case .patchCountMismatch(let patches, let grid):
            return "llava_onevision2: \(patches) patches but grid_thw covers \(grid)"
        case .missingImageGrid:
            return "llava_onevision2: image input has no grid (THW) information"
        case .videoNotImplemented:
            return "llava_onevision2: video input is not supported; pass frames as images"
        case .emptyPrompt:
            return "llava_onevision2: prompt has no tokens"
        case .batchSizeNotSupported(let batch):
            return "llava_onevision2: batch size \(batch) is not supported; use 1"
        }
    }
}

/// Qwen3-style text backbone from `text_config`.
public struct SharpAILlavaOnevision2TextConfiguration: Decodable, Sendable {
    public let modelType: String
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
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let decodedType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3"
        let hidden = try container.decode(Int.self, forKey: .hiddenSize)
        let heads = try container.decode(Int.self, forKey: .attentionHeads)

        modelType = decodedType
        hiddenSize = hidden
        hiddenLayers = try container.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = heads
        kvHeads = try container.decode(Int.self, forKey: .kvHeads)
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (heads > 0 ? hidden / heads : 0)
        vocabularySize = try container.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try container.decode(Float.self, forKey: .rmsNormEps)

        let rope = try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        let flatTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        guard let theta = rope?.ropeTheta ?? flatTheta else {
            throw DecodingError.dataCorruptedError(
                forKey: .ropeTheta, in: container,
                debugDescription: "rope_theta is missing from rope_parameters and text_config")
        }
        ropeTheta = theta

        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// `onevision_encoder` vision tower and projector fields from `vision_config`.
public struct SharpAILlavaOnevision2VisionConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let spatialMergeSize: Int
    public let outHiddenSize: Int
    public let textHiddenSize: Int
    public let frameWindowSize: Int
    public let tokensPerSecond: Int
    public let hiddenAct: String
    public let layerNormType: String
    public let layerNormEps: Float
    public let useHead: Bool
    /// Optional in the checkpoint; the reference `VisionRotaryEmbedding` default is 10000.0.
    public let ropeTheta: Float?

    /// RoPE base used by the vision tower (reference default when the key is absent).
    public var effectiveRopeTheta: Float { ropeTheta ?? 10000.0 }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case spatialMergeSize = "spatial_merge_size"
        case outHiddenSize = "out_hidden_size"
        case textHiddenSize = "text_hidden_size"
        case frameWindowSize = "frame_windows_size"
        case tokensPerSecond = "tokens_per_second"
        case hiddenAct = "hidden_act"
        case layerNormType = "layer_norm_type"
        case layerNormEps = "layer_norm_eps"
        case useHead = "use_head"
        case ropeTheta = "rope_theta"
    }
}

/// Top-level `config.json` for `model_type: llava_onevision2`.
public struct SharpAILlavaOnevision2Configuration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]
    public let textConfiguration: SharpAILlavaOnevision2TextConfiguration
    public let visionConfiguration: SharpAILlavaOnevision2VisionConfiguration
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "llava_onevision2"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        textConfiguration = try container.decode(
            SharpAILlavaOnevision2TextConfiguration.self, forKey: .textConfig)
        visionConfiguration = try container.decode(
            SharpAILlavaOnevision2VisionConfiguration.self, forKey: .visionConfig)
        imageTokenId = try container.decode(Int.self, forKey: .imageTokenId)
        videoTokenId = try container.decode(Int.self, forKey: .videoTokenId)
        visionStartTokenId = try container.decode(Int.self, forKey: .visionStartTokenId)
        visionEndTokenId = try container.decode(Int.self, forKey: .visionEndTokenId)
    }

    /// Patch grid side for one square image, e.g. 448 / 14 = 32.
    public var visionGridSide: Int {
        visionConfiguration.imageSize / visionConfiguration.patchSize
    }

    /// Tokens left after the spatial merge, per square image. Valid after validation.
    /// One merged token per `spatial_merge_size`² patch group, as in the reference merger.
    public var imageTokensPerImage: Int {
        let side = visionGridSide / visionConfiguration.spatialMergeSize
        return side * side
    }

    public func validateModelConfiguration() throws {
        guard modelType == "llava_onevision2" else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: unexpected model_type \(modelType)")
        }

        let text = textConfiguration
        let textDims = [
            text.hiddenSize, text.hiddenLayers, text.intermediateSize, text.attentionHeads,
            text.kvHeads, text.headDim, text.vocabularySize,
        ]
        guard textDims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: text dimensions must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard !text.tieWordEmbeddings else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: tie_word_embeddings is not supported (the 8B checkpoint is untied)")
        }

        let vision = visionConfiguration
        guard vision.attentionHeads > 0, vision.hiddenSize % vision.attentionHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision hidden_size must be divisible by num_attention_heads")
        }
        // Reference VisionRotaryEmbedding: head_dim % 16 == 0 and head_dim // 2 % 16 == 0.
        guard (vision.hiddenSize / vision.attentionHeads) % 32 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision head_dim must be divisible by 32 for the 4:6:6 RoPE split")
        }
        guard !vision.useHead else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision use_head=true (pooling head) is not implemented")
        }
        guard vision.numChannels > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision num_channels must be positive")
        }
        // The vision tower builds LayerNorm and the exact GELU only.
        guard vision.layerNormType == "layer_norm" else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision layer_norm_type \(vision.layerNormType) is not supported")
        }
        guard vision.hiddenAct == "gelu" else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision hidden_act \(vision.hiddenAct) is not supported")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision image_size must be a multiple of patch_size")
        }
        guard vision.spatialMergeSize > 0, visionGridSide % vision.spatialMergeSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: patch grid must be divisible by spatial_merge_size")
        }
        guard vision.outHiddenSize == text.hiddenSize else {
            throw ModelFactoryError.invalidConfiguration(
                "llava_onevision2: vision out_hidden_size must equal text hidden_size")
        }
    }
}
