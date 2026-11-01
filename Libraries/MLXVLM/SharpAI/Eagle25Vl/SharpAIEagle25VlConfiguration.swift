// SharpAI fork overlay: eagle_2_5_vl configuration. Not part of upstream mlx-swift-lm.
//
// Keys follow vLLM's model_executor/models/eagle2_5_vl.py (NVIDIA Eagle2.5-VL, SigLIP vision,
// Qwen2 text, InternVL-style top-level keys).
// UNVERIFIED: the HF checkpoint is gated, so no config.json was read. The `img_context_token_id`
// key is an assumption; vLLM takes the image token id from the tokenizer instead.

import Foundation
import MLXLMCommon

public enum SharpAIEagle25VlError: LocalizedError {
    case unsupportedActivation(String)
    case imageTokenIdMissing
    case imageTokenCountMismatch(expected: Int, actual: Int)
    case batchedImageUnsupported(Int)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedActivation(let name):
            return "eagle_2_5_vl: unsupported vision hidden_act \(name)"
        case .imageTokenIdMissing:
            return "eagle_2_5_vl: images need img_context_token_id in config.json (UNVERIFIED key)"
        case .imageTokenCountMismatch(let expected, let actual):
            return
                "eagle_2_5_vl: prompt has \(expected) image tokens, image features have \(actual) rows"
        case .batchedImageUnsupported(let batch):
            return "eagle_2_5_vl: image inputs need batch size 1, got \(batch)"
        case .duplicateWeightKey(let key):
            return "eagle_2_5_vl: duplicate weight key after sanitize: \(key)"
        }
    }
}

/// SigLIP vision tower settings (`vision_config`).
public struct SharpAIEagle25VlVisionConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let imageSize: Int
    public let patchSize: Int
    public let channels: Int
    public let layerNormEps: Float
    public let hiddenAct: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case channels = "num_channels"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        imageSize = try c.decode(Int.self, forKey: .imageSize)
        patchSize = try c.decode(Int.self, forKey: .patchSize)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
    }
}

/// Qwen2 text settings (`text_config`).
public struct SharpAIEagle25VlTextConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
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
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Top-level eagle_2_5_vl configuration.
public struct SharpAIEagle25VlConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "eagle_2_5_vl"

    public let modelType: String
    public let visionConfiguration: SharpAIEagle25VlVisionConfiguration
    public let textConfiguration: SharpAIEagle25VlTextConfiguration
    public let downsampleRatio: Float
    public let selectLayer: Int
    public let forceImageSize: Int?
    public let dynamicImageSize: Bool
    public let useThumbnail: Bool
    public let minDynamicPatch: Int
    public let maxDynamicPatch: Int
    /// Token id of `<IMG_CONTEXT>`, where image features are written.
    public let imageTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case downsampleRatio = "downsample_ratio"
        case selectLayer = "select_layer"
        case forceImageSize = "force_image_size"
        case dynamicImageSize = "dynamic_image_size"
        case useThumbnail = "use_thumbnail"
        case minDynamicPatch = "min_dynamic_patch"
        case maxDynamicPatch = "max_dynamic_patch"
        case imageTokenId = "img_context_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType
        visionConfiguration = try c.decode(
            SharpAIEagle25VlVisionConfiguration.self, forKey: .visionConfiguration)
        textConfiguration = try c.decode(
            SharpAIEagle25VlTextConfiguration.self, forKey: .textConfiguration)
        downsampleRatio = try c.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        selectLayer = try c.decodeIfPresent(Int.self, forKey: .selectLayer) ?? -1
        forceImageSize = try c.decodeIfPresent(Int.self, forKey: .forceImageSize)
        dynamicImageSize = try c.decodeIfPresent(Bool.self, forKey: .dynamicImageSize) ?? true
        useThumbnail = try c.decodeIfPresent(Bool.self, forKey: .useThumbnail) ?? true
        minDynamicPatch = try c.decodeIfPresent(Int.self, forKey: .minDynamicPatch) ?? 1
        maxDynamicPatch = try c.decodeIfPresent(Int.self, forKey: .maxDynamicPatch) ?? 12
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId)
    }

    /// Vision encoder layers kept, as vLLM's `num_hidden_layers_override`.
    public var visionLayerCount: Int {
        let total = visionConfiguration.hiddenLayers
        return selectLayer < 0 ? total + selectLayer + 1 : selectLayer + 1
    }

    /// `post_layernorm` exists only when every encoder layer is kept.
    public var appliesVisionPostLayerNorm: Bool {
        visionLayerCount == visionConfiguration.hiddenLayers
    }

    /// Side length of one tile in pixels.
    public var effectiveImageSize: Int {
        forceImageSize ?? visionConfiguration.imageSize
    }

    /// Pixel-shuffle factor, `1 / downsample_ratio`.
    public var pixelShuffleFactor: Int {
        Int((1 / downsampleRatio).rounded())
    }

    /// Image tokens per tile after pixel shuffle.
    public var imageSeqLength: Int {
        let grid = effectiveImageSize / visionConfiguration.patchSize
        let side = grid / pixelShuffleFactor
        return side * side
    }

    /// Input width of the projector MLP.
    public var projectorInputSize: Int {
        visionConfiguration.hiddenSize * pixelShuffleFactor * pixelShuffleFactor
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        let vision = visionConfiguration
        let text = textConfiguration

        guard vision.hiddenSize > 0, vision.attentionHeads > 0,
            vision.hiddenSize % vision.attentionHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: vision hidden_size must divide by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: vision image_size must be a multiple of patch_size")
        }
        guard vision.hiddenAct == "gelu_pytorch_tanh" || vision.hiddenAct == "gelu" else {
            throw SharpAIEagle25VlError.unsupportedActivation(vision.hiddenAct)
        }
        guard effectiveImageSize == vision.imageSize else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: force_image_size \(effectiveImageSize) differs from vision image_size "
                    + "\(vision.imageSize); position interpolation is not implemented")
        }

        guard downsampleRatio > 0, downsampleRatio <= 1,
            abs(1 / downsampleRatio - Float(pixelShuffleFactor)) < 1e-4
        else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: downsample_ratio must be 1 / integer, got \(downsampleRatio)")
        }
        let grid = vision.imageSize / vision.patchSize
        guard grid % pixelShuffleFactor == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: patch grid \(grid) is not divisible by pixel shuffle factor")
        }

        guard (1 ... vision.hiddenLayers).contains(visionLayerCount) else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: select_layer \(selectLayer) keeps \(visionLayerCount) of "
                    + "\(vision.hiddenLayers) vision layers")
        }

        guard text.attentionHeads > 0, text.hiddenSize % text.attentionHeads == 0,
            text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0, text.vocabularySize > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "eagle_2_5_vl: text heads and vocab_size are inconsistent")
        }
    }
}
