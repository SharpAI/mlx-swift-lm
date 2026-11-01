// SharpAI fork overlay: qianfan_ocr configuration. Not part of upstream mlx-swift-lm.
// Field names follow baidu/Qianfan-OCR config.json: InternViT vision tower, Qwen3 text decoder.
// Defaults match transformers' QianfanOCRConfig where the Hub file omits a key.

import Foundation
import MLXLMCommon

public enum SharpAIQianfanOcrError: LocalizedError {
    case unsupported(String)
    case invalidInput(String)
    case imageTokenMismatch(placeholders: Int, features: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "qianfan_ocr unsupported configuration: \(detail)"
        case .invalidInput(let detail):
            return "qianfan_ocr invalid input: \(detail)"
        case .imageTokenMismatch(let placeholders, let features):
            return "qianfan_ocr has \(placeholders) image tokens, \(features) image embeddings."
        }
    }
}

/// `vision_config` (`model_type: qianfan_ocr_vision`): InternViT with learned absolute positions.
public struct SharpAIQianfanOcrVisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let intermediateSize: Int
    public let hiddenAct: String
    public let layerNormEps: Float
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let normType: String
    public let useMeanPooling: Bool
    public let qkvBias: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case normType = "norm_type"
        case useMeanPooling = "use_mean_pooling"
        case qkvBias = "qkv_bias"
    }

    public init(
        hiddenSize: Int = 1024,
        hiddenLayers: Int = 24,
        attentionHeads: Int = 16,
        intermediateSize: Int = 4096,
        hiddenAct: String = "gelu",
        layerNormEps: Float = 1e-6,
        imageSize: Int = 448,
        patchSize: Int = 14,
        numChannels: Int = 3,
        normType: String = "layer_norm",
        useMeanPooling: Bool = true,
        qkvBias: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.intermediateSize = intermediateSize
        self.hiddenAct = hiddenAct
        self.layerNormEps = layerNormEps
        self.imageSize = imageSize
        self.patchSize = patchSize
        self.numChannels = numChannels
        self.normType = normType
        self.useMeanPooling = useMeanPooling
        self.qkvBias = qkvBias
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIQianfanOcrVisionConfiguration()
        self.init(
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? defaults.hiddenSize,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? defaults.hiddenLayers,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? defaults.attentionHeads,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? defaults.intermediateSize,
            hiddenAct: try container.decodeIfPresent(String.self, forKey: .hiddenAct)
                ?? defaults.hiddenAct,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? defaults.layerNormEps,
            imageSize: try container.decodeIfPresent(Int.self, forKey: .imageSize)
                ?? defaults.imageSize,
            patchSize: try container.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? defaults.patchSize,
            numChannels: try container.decodeIfPresent(Int.self, forKey: .numChannels)
                ?? defaults.numChannels,
            normType: try container.decodeIfPresent(String.self, forKey: .normType)
                ?? defaults.normType,
            useMeanPooling: try container.decodeIfPresent(Bool.self, forKey: .useMeanPooling)
                ?? defaults.useMeanPooling,
            qkvBias: try container.decodeIfPresent(Bool.self, forKey: .qkvBias)
                ?? defaults.qkvBias
        )
    }
}

/// `text_config` (`model_type: qwen3`): the Qwen3 decoder that consumes the image embeddings.
/// UNVERIFIED: `rope_scaling` is not decoded. The Hub config sets it to null.
public struct SharpAIQianfanOcrTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(
        modelType: String = "qwen3",
        hiddenSize: Int = 2560,
        hiddenLayers: Int = 36,
        intermediateSize: Int = 9728,
        attentionHeads: Int = 32,
        kvHeads: Int? = nil,
        headDim: Int? = nil,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 151_936,
        ropeTheta: Float = 1_000_000,
        attentionBias: Bool = false,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.headDim = headDim ?? 128
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.attentionBias = attentionBias
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIQianfanOcrTextConfiguration()
        let heads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? defaults.attentionHeads
        let hidden =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? defaults.hiddenSize
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? defaults.modelType,
            hiddenSize: hidden,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? defaults.hiddenLayers,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? defaults.intermediateSize,
            attentionHeads: heads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? defaults.rmsNormEps,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? defaults.vocabularySize,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? defaults.ropeTheta,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? defaults.attentionBias,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self, forKey: .tieWordEmbeddings) ?? defaults.tieWordEmbeddings
        )
    }
}

/// Top-level `config.json` for `model_type: qianfan_ocr`.
public struct SharpAIQianfanOcrConfiguration: Codable, Sendable {
    public let modelType: String
    public let visionConfiguration: SharpAIQianfanOcrVisionConfiguration
    public let textConfiguration: SharpAIQianfanOcrTextConfiguration
    /// Placeholder id that the processor expands to one run of image tokens per image.
    public let imageTokenId: Int
    /// Pixel-shuffle ratio. 0.5 merges 2x2 patch tokens into one token.
    public let downsampleRatio: Float
    public let projectorHiddenAct: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case imageTokenId = "image_token_id"
        case downsampleRatio = "downsample_ratio"
        case projectorHiddenAct = "projector_hidden_act"
    }

    public init(
        modelType: String = "qianfan_ocr",
        visionConfiguration: SharpAIQianfanOcrVisionConfiguration = .init(),
        textConfiguration: SharpAIQianfanOcrTextConfiguration = .init(),
        imageTokenId: Int = 151_671,
        downsampleRatio: Float = 0.5,
        projectorHiddenAct: String = "gelu"
    ) {
        self.modelType = modelType
        self.visionConfiguration = visionConfiguration
        self.textConfiguration = textConfiguration
        self.imageTokenId = imageTokenId
        self.downsampleRatio = downsampleRatio
        self.projectorHiddenAct = projectorHiddenAct
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = SharpAIQianfanOcrConfiguration()
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? defaults.modelType,
            visionConfiguration: try container.decodeIfPresent(
                SharpAIQianfanOcrVisionConfiguration.self, forKey: .visionConfiguration)
                ?? defaults.visionConfiguration,
            textConfiguration: try container.decodeIfPresent(
                SharpAIQianfanOcrTextConfiguration.self, forKey: .textConfiguration)
                ?? defaults.textConfiguration,
            imageTokenId: try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
                ?? defaults.imageTokenId,
            downsampleRatio: try container.decodeIfPresent(Float.self, forKey: .downsampleRatio)
                ?? defaults.downsampleRatio,
            projectorHiddenAct: try container.decodeIfPresent(
                String.self, forKey: .projectorHiddenAct) ?? defaults.projectorHiddenAct
        )
    }

    /// Integer factor per spatial axis, `1 / downsampleRatio` (2 for the Hub checkpoint).
    public var downsampleFactor: Int {
        Int((1 / downsampleRatio).rounded())
    }
}

extension SharpAIQianfanOcrConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        let vision = visionConfiguration
        let text = textConfiguration

        guard modelType == "qianfan_ocr" else {
            throw SharpAIQianfanOcrError.unsupported("model_type \(modelType)")
        }
        guard projectorHiddenAct == "gelu" else {
            throw SharpAIQianfanOcrError.unsupported("projector_hidden_act \(projectorHiddenAct)")
        }

        guard vision.hiddenAct == "gelu" else {
            throw SharpAIQianfanOcrError.unsupported("vision hidden_act \(vision.hiddenAct)")
        }
        guard vision.normType == "layer_norm" else {
            throw SharpAIQianfanOcrError.unsupported("vision norm_type \(vision.normType)")
        }
        // Without mean pooling the reference adds a final LayerNorm. This overlay does not.
        guard vision.useMeanPooling else {
            throw SharpAIQianfanOcrError.unsupported("vision use_mean_pooling = false")
        }
        guard vision.hiddenLayers > 0, vision.attentionHeads > 0,
            vision.hiddenSize % vision.attentionHeads == 0
        else {
            throw SharpAIQianfanOcrError.unsupported(
                "vision hidden_size must be divisible by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw SharpAIQianfanOcrError.unsupported("vision image_size must divide by patch_size")
        }

        guard downsampleRatio > 0 else {
            throw SharpAIQianfanOcrError.unsupported("downsample_ratio must be positive")
        }
        let exactFactor = 1 / downsampleRatio
        guard abs(exactFactor - Float(downsampleFactor)) < 1e-4, downsampleFactor >= 1 else {
            throw SharpAIQianfanOcrError.unsupported(
                "downsample_ratio \(downsampleRatio) must be 1 / integer")
        }
        let grid = vision.imageSize / vision.patchSize
        guard grid % downsampleFactor == 0 else {
            throw SharpAIQianfanOcrError.unsupported(
                "patch grid \(grid) must divide by the downsample factor \(downsampleFactor)")
        }

        guard text.modelType == "qwen3" else {
            throw SharpAIQianfanOcrError.unsupported("text model_type \(text.modelType)")
        }
        guard text.hiddenLayers > 0, text.attentionHeads > 0, text.kvHeads > 0,
            text.headDim > 0
        else {
            throw SharpAIQianfanOcrError.unsupported("text layer and head counts must be positive")
        }
        guard text.attentionHeads % text.kvHeads == 0 else {
            throw SharpAIQianfanOcrError.unsupported(
                "text num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
