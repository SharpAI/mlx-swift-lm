// SharpAI fork overlay: moondream3 configuration. Not part of upstream mlx-swift-lm.
//
// Defaults and key names come from vLLM's transformers_utils/configs/moondream3.py. The nested
// `config` object is what vLLM reads from config.json.
// UNVERIFIED: no moondream3 config.json was read. Defaults are the vLLM values.

import Foundation
import MLXLMCommon

public enum SharpAIMoondream3Error: LocalizedError {
    case singleImageOnly(Int)
    case imageShape([Int])
    case missingTiling
    case tilingMismatch(crops: Int, tiles: Int)
    case batchedInputUnsupported(Int)
    case promptTooShort(tokens: Int, prefix: Int)
    case imagePlaceholderUnavailable
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .singleImageOnly(let count):
            return "moondream3: one image per prompt is supported, got \(count)"
        case .imageShape(let shape):
            return "moondream3: image pixels must be [crops, 3, crop, crop], got \(shape)"
        case .missingTiling:
            return "moondream3: multi-crop images need tiling (rows, cols) in ProcessedImage.frames"
        case .tilingMismatch(let crops, let tiles):
            return "moondream3: \(crops) local crops do not match a tiling of \(tiles) tiles"
        case .batchedInputUnsupported(let batch):
            return "moondream3: image inputs need batch size 1, got \(batch)"
        case .promptTooShort(let tokens, let prefix):
            return "moondream3: prompt has \(tokens) tokens, image prefix needs \(prefix)"
        case .imagePlaceholderUnavailable:
            return "moondream3: tokenizer must encode <image> as a single token"
        case .duplicateWeightKey(let key):
            return "moondream3: duplicate weight key after sanitize: \(key)"
        }
    }
}

/// Text decoder settings (`config.text`).
public struct SharpAIMoondream3TextConfiguration: Decodable, Sendable {
    public let dim: Int
    public let ffDim: Int
    public let layers: Int
    public let vocabularySize: Int
    public let maxContext: Int
    public let heads: Int
    public let kvHeads: Int
    /// Length of the bidirectional image prefix in HF. Not applied: see the model file.
    public let prefixAttn: Int
    public let ropeTheta: Float
    /// First layer index that uses the MoE block.
    public let moeStartLayer: Int
    public let moeExperts: Int
    public let moeExpertsPerToken: Int
    public let moeExpertInnerDim: Int

    public init(
        dim: Int = 2048, ffDim: Int = 8192, layers: Int = 24, vocabularySize: Int = 51200,
        maxContext: Int = 4096, heads: Int = 32, kvHeads: Int = 32, prefixAttn: Int = 730,
        ropeTheta: Float = 1_500_000, moeStartLayer: Int = 4, moeExperts: Int = 64,
        moeExpertsPerToken: Int = 8, moeExpertInnerDim: Int = 1024
    ) {
        self.dim = dim
        self.ffDim = ffDim
        self.layers = layers
        self.vocabularySize = vocabularySize
        self.maxContext = maxContext
        self.heads = heads
        self.kvHeads = kvHeads
        self.prefixAttn = prefixAttn
        self.ropeTheta = ropeTheta
        self.moeStartLayer = moeStartLayer
        self.moeExperts = moeExperts
        self.moeExpertsPerToken = moeExpertsPerToken
        self.moeExpertInnerDim = moeExpertInnerDim
    }

    enum CodingKeys: String, CodingKey {
        case dim, layers = "n_layers", ffDim = "ff_dim", vocabularySize = "vocab_size"
        case maxContext = "max_context", heads = "n_heads", kvHeads = "n_kv_heads"
        case prefixAttn = "prefix_attn", ropeTheta = "rope_theta", moe
    }

    enum MoEKeys: String, CodingKey {
        case startLayer = "start_layer", experts = "n_experts"
        case expertsPerToken = "n_experts_per_tok", expertInnerDim = "expert_inner_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SharpAIMoondream3TextConfiguration()
        dim = try c.decodeIfPresent(Int.self, forKey: .dim) ?? d.dim
        ffDim = try c.decodeIfPresent(Int.self, forKey: .ffDim) ?? d.ffDim
        layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? d.layers
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? d.vocabularySize
        maxContext = try c.decodeIfPresent(Int.self, forKey: .maxContext) ?? d.maxContext
        heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? d.heads
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? d.kvHeads
        prefixAttn = try c.decodeIfPresent(Int.self, forKey: .prefixAttn) ?? d.prefixAttn
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? d.ropeTheta

        if c.contains(.moe) {
            let m = try c.nestedContainer(keyedBy: MoEKeys.self, forKey: .moe)
            moeStartLayer = try m.decodeIfPresent(Int.self, forKey: .startLayer) ?? d.moeStartLayer
            moeExperts = try m.decodeIfPresent(Int.self, forKey: .experts) ?? d.moeExperts
            moeExpertsPerToken =
                try m.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? d.moeExpertsPerToken
            moeExpertInnerDim =
                try m.decodeIfPresent(Int.self, forKey: .expertInnerDim) ?? d.moeExpertInnerDim
        } else {
            moeStartLayer = d.moeStartLayer
            moeExperts = d.moeExperts
            moeExpertsPerToken = d.moeExpertsPerToken
            moeExpertInnerDim = d.moeExpertInnerDim
        }
    }
}

/// Vision encoder and projector settings (`config.vision`).
public struct SharpAIMoondream3VisionConfiguration: Decodable, Sendable {
    public let encDim: Int
    public let patchSize: Int
    /// Encoder depth. vLLM also uses it as the pooled feature grid side.
    public let encLayers: Int
    public let encFFDim: Int
    public let encHeads: Int
    public let projInnerDim: Int
    public let cropSize: Int
    public let maxCrops: Int
    public let overlapMargin: Int

    public init(
        encDim: Int = 1152, patchSize: Int = 14, encLayers: Int = 27, encFFDim: Int = 4304,
        encHeads: Int = 16, projInnerDim: Int = 8192, cropSize: Int = 378, maxCrops: Int = 12,
        overlapMargin: Int = 4
    ) {
        self.encDim = encDim
        self.patchSize = patchSize
        self.encLayers = encLayers
        self.encFFDim = encFFDim
        self.encHeads = encHeads
        self.projInnerDim = projInnerDim
        self.cropSize = cropSize
        self.maxCrops = maxCrops
        self.overlapMargin = overlapMargin
    }

    enum CodingKeys: String, CodingKey {
        case encDim = "enc_dim", patchSize = "enc_patch_size", encLayers = "enc_n_layers"
        case encFFDim = "enc_ff_dim", encHeads = "enc_n_heads", projInnerDim = "proj_inner_dim"
        case cropSize = "crop_size", maxCrops = "max_crops", overlapMargin = "overlap_margin"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = SharpAIMoondream3VisionConfiguration()
        encDim = try c.decodeIfPresent(Int.self, forKey: .encDim) ?? d.encDim
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? d.patchSize
        encLayers = try c.decodeIfPresent(Int.self, forKey: .encLayers) ?? d.encLayers
        encFFDim = try c.decodeIfPresent(Int.self, forKey: .encFFDim) ?? d.encFFDim
        encHeads = try c.decodeIfPresent(Int.self, forKey: .encHeads) ?? d.encHeads
        projInnerDim = try c.decodeIfPresent(Int.self, forKey: .projInnerDim) ?? d.projInnerDim
        cropSize = try c.decodeIfPresent(Int.self, forKey: .cropSize) ?? d.cropSize
        maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? d.maxCrops
        overlapMargin = try c.decodeIfPresent(Int.self, forKey: .overlapMargin) ?? d.overlapMargin
    }

    /// Feature grid side for one crop (27 for the released 378 px crop).
    public var gridSize: Int { cropSize / patchSize }
}

/// Top-level moondream3 configuration. Decodes the nested `config` object.
public struct SharpAIMoondream3Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "moondream3"
    /// Token 0 (`<|endoftext|>`) starts every prompt and fills the image BOS slot.
    public static let bosTokenId = 0

    public let modelType: String
    public let textConfiguration: SharpAIMoondream3TextConfiguration
    public let visionConfiguration: SharpAIMoondream3VisionConfiguration
    /// Logit suppressed during generation (`<|md_reserved_2|>`).
    public let answerTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case config
    }

    enum ConfigKeys: String, CodingKey {
        case text, vision, tokenizer
    }

    enum TokenizerKeys: String, CodingKey {
        case answerId = "answer_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType

        if c.contains(.config) {
            let inner = try c.nestedContainer(keyedBy: ConfigKeys.self, forKey: .config)
            textConfiguration =
                try inner.decodeIfPresent(SharpAIMoondream3TextConfiguration.self, forKey: .text)
                ?? SharpAIMoondream3TextConfiguration()
            visionConfiguration =
                try inner.decodeIfPresent(
                    SharpAIMoondream3VisionConfiguration.self, forKey: .vision)
                ?? SharpAIMoondream3VisionConfiguration()
            let tokenizer = try? inner.nestedContainer(
                keyedBy: TokenizerKeys.self, forKey: .tokenizer)
            answerTokenId = try tokenizer?.decodeIfPresent(Int.self, forKey: .answerId) ?? 3
        } else {
            textConfiguration = SharpAIMoondream3TextConfiguration()
            visionConfiguration = SharpAIMoondream3VisionConfiguration()
            answerTokenId = 3
        }
    }

    /// Tokens an image occupies in the prompt: BOS plus one row per pooled vision feature.
    public var imagePrefixLength: Int {
        1 + visionConfiguration.encLayers * visionConfiguration.encLayers
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        let t = textConfiguration
        guard t.layers > 0, t.vocabularySize > 0, t.heads > 0, t.dim % t.heads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: text dim must divide by n_heads, and layers and vocab must be positive")
        }
        // UNVERIFIED: vLLM only implements MHA here, so grouped KV heads are rejected.
        guard t.kvHeads == t.heads else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: only n_kv_heads == n_heads is supported")
        }
        guard (t.dim / t.heads) % 2 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: head dim must be even for half-width RoPE")
        }
        guard t.moeStartLayer >= 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: moe start_layer must not be negative")
        }
        if t.moeStartLayer < t.layers {
            guard t.moeExpertsPerToken > 0, t.moeExperts >= t.moeExpertsPerToken else {
                throw ModelFactoryError.invalidConfiguration(
                    "moondream3: n_experts_per_tok must be between 1 and n_experts")
            }
        }

        let v = visionConfiguration
        guard v.encHeads > 0, v.encDim % v.encHeads == 0, v.projInnerDim > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: enc_dim must divide by enc_n_heads")
        }
        guard v.patchSize > 0, v.cropSize % v.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: crop_size must be a multiple of enc_patch_size")
        }
        // Pooling is an identity in this port, so the grid must already equal the pooled side.
        guard v.encLayers == v.gridSize else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: enc_n_layers (\(v.encLayers)) must equal the crop grid (\(v.gridSize))")
        }
        guard v.overlapMargin >= 0, v.gridSize > 2 * v.overlapMargin, v.maxCrops >= 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "moondream3: overlap_margin must leave a non-empty crop window")
        }
    }
}
