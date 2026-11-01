// SharpAI fork overlay: PaddleOCR-VL configuration. Not part of upstream mlx-swift-lm.
//
// Field names follow the PaddleOCR-VL config.json (model_type "paddleocr_vl").
// Defaults follow the repo fixture scripts/conformance/offline/fixtures/paddleocr_vl/
// PaddlePaddle__PaddleOCR-VL.json where it has a value. Fields the fixture lacks are UNVERIFIED.

import Foundation
import MLXLMCommon

public enum SharpAIPaddleocrVlError: LocalizedError {
    case invalidConfiguration(String)
    case batchUnsupported(Int)
    case invalidPixelShape([Int])
    case invalidImageGrid(height: Int, width: Int, mergeSize: Int)
    case unsupportedTemporalFrames(Int)
    case placeholderMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "paddleocr_vl: invalid configuration: \(detail)"
        case .batchUnsupported(let batch):
            return "paddleocr_vl: batch size must be 1, got \(batch)."
        case .invalidPixelShape(let shape):
            return "paddleocr_vl: pixels must be (patches, P, P, 3), got \(shape)."
        case .invalidImageGrid(let height, let width, let mergeSize):
            return "paddleocr_vl: patch grid \(height)x\(width) must divide by \(mergeSize)."
        case .unsupportedTemporalFrames(let frames):
            return "paddleocr_vl: only single-frame images are supported, got t=\(frames)."
        case .placeholderMismatch(let expected, let actual):
            return
                "paddleocr_vl: prompt has \(expected) image placeholders, "
                + "but the vision tower produced \(actual) tokens."
        }
    }
}

public struct SharpAIPaddleocrVlTextConfiguration: Decodable, Sendable {
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    /// Sizes of the t, h and w sections over the half-dim rotary frequencies.
    public let mropeSection: [Int]
    public let tieWordEmbeddings: Bool
    public let useBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useBias = "use_bias"
    }

    private struct RopeScaling: Decodable {
        let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 103_424
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 18
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        let scaling = try container.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling)
        mropeSection = scaling?.mropeSection ?? [16, 24, 24]
        tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
    }
}

public struct SharpAIPaddleocrVlVisionConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let numChannels: Int
    public let imageSize: Int
    public let patchSize: Int
    public let spatialMergeSize: Int
    public let layerNormEps: Float

    /// Patches per side of the pretrained position grid.
    public var patchesPerSide: Int { imageSize / patchSize }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case spatialMergeSize = "spatial_merge_size"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        spatialMergeSize =
            try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

public struct SharpAIPaddleocrVlConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public let textConfiguration: SharpAIPaddleocrVlTextConfiguration
    public let visionConfiguration: SharpAIPaddleocrVlVisionConfiguration
    /// Placeholder id for image patches in the prompt.
    public let imageTokenId: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenId = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // Text fields are top-level in the real config. A nested text_config also works.
        if let nested = try container.decodeIfPresent(
            SharpAIPaddleocrVlTextConfiguration.self, forKey: .textConfiguration)
        {
            textConfiguration = nested
        } else {
            textConfiguration = try SharpAIPaddleocrVlTextConfiguration(from: decoder)
        }
        visionConfiguration = try container.decode(
            SharpAIPaddleocrVlVisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 100_295
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let vision = visionConfiguration

        guard text.attentionHeads > 0, text.kvHeads > 0,
            text.attentionHeads % text.kvHeads == 0
        else {
            throw SharpAIPaddleocrVlError.invalidConfiguration(
                "attention heads must be a multiple of kv heads")
        }
        guard text.mropeSection.count == 3, text.mropeSection.allSatisfy({ $0 > 0 }),
            text.mropeSection.reduce(0, +) * 2 == text.headDim
        else {
            throw SharpAIPaddleocrVlError.invalidConfiguration(
                "mrope_section needs three positive sizes that sum to head_dim / 2")
        }
        guard vision.attentionHeads > 0, vision.hiddenSize % vision.attentionHeads == 0 else {
            throw SharpAIPaddleocrVlError.invalidConfiguration(
                "vision hidden size must divide by vision heads")
        }
        // image_size need not divide by patch_size: 384 / 14 gives the 27x27 position grid.
        guard vision.patchSize > 0, vision.patchesPerSide > 0 else {
            throw SharpAIPaddleocrVlError.invalidConfiguration(
                "image_size must be at least patch_size")
        }
        guard vision.spatialMergeSize > 0 else {
            throw SharpAIPaddleocrVlError.invalidConfiguration(
                "spatial_merge_size must be positive")
        }
    }
}
