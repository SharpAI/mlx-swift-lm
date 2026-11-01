// SharpAI fork overlay: phi4-siglip configuration. Not part of upstream mlx-swift-lm.
//
// Top-level keys hold a Phi-3-style text decoder. `vision_config` holds a SigLIP2 NaFlex tower.
// UNVERIFIED: no checkpoint or modeling source was read. The layout follows the offline fixture
// scripts/conformance/offline/fixtures/phi4-siglip/microsoft__Phi-4-reasoning-vision-15B.json.
// Keys absent from that fixture (vision patch size, position count, image token index) use the
// SigLIP2 defaults or stay optional.

import Foundation
import MLXLMCommon

public enum SharpAIPhi4SiglipError: LocalizedError {
    case unsupportedRopeScaling
    case imageTokenIndexMissing
    case imagePlaceholderMismatch(expected: Int, actual: Int)
    case invalidImageFrame
    case imageGridMismatch(expected: Int, actual: Int)
    case missingImageGrid
    case batchedImageUnsupported(Int)
    case duplicateWeightKey(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedRopeScaling:
            return "phi4-siglip: rope_scaling is not implemented"
        case .imageTokenIndexMissing:
            return "phi4-siglip: images need image_token_index in config.json"
        case .imagePlaceholderMismatch(let expected, let actual):
            return "phi4-siglip: expected \(expected) image placeholders, found \(actual)"
        case .invalidImageFrame:
            return "phi4-siglip: each image needs one frame with t == 1 and a non-empty grid"
        case .imageGridMismatch(let expected, let actual):
            return "phi4-siglip: frames cover \(expected) patch rows, pixels have \(actual)"
        case .missingImageGrid:
            return "phi4-siglip: image input has no patch grid"
        case .batchedImageUnsupported(let batch):
            return "phi4-siglip: image inputs need batch size 1, got \(batch)"
        case .duplicateWeightKey(let key):
            return "phi4-siglip: duplicate weight key after sanitize: \(key)"
        }
    }
}

/// SigLIP2 NaFlex vision tower settings (`vision_config`).
public struct SharpAIPhi4SiglipVisionConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let channels: Int
    public let patchSize: Int
    /// Size of the learned position table, a square grid of positions.
    public let positionCount: Int
    public let layerNormEps: Float
    public let hiddenAct: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case channels = "num_channels"
        case patchSize = "patch_size"
        case positionCount = "num_patches"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        positionCount = try c.decodeIfPresent(Int.self, forKey: .positionCount) ?? 256
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
    }
}

/// Top-level phi4-siglip configuration. The text fields sit at the top level.
public struct SharpAIPhi4SiglipConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "phi4-siglip"

    public let modelType: String
    public let visionConfiguration: SharpAIPhi4SiglipVisionConfiguration

    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    /// True when `rope_scaling` is set. Scaled RoPE is not implemented, so validation rejects it.
    public let hasRopeScaling: Bool
    public let tieWordEmbeddings: Bool

    /// Input width of the projector. Must equal the vision hidden size.
    public let mmHiddenSize: Int
    public let mmProjectorType: String

    /// NaFlex patch budget per image. `min_num_patches` is parsed but not enforced (UNVERIFIED).
    public let maxNumPatches: Int
    public let minNumPatches: Int

    /// Token id that image features replace, when the config carries one. The released
    /// checkpoint config has no such key; use `resolvedImageTokenIndex`.
    public let imageTokenIndex: Int?

    /// Phi-4 tokenizer id of `<|dummy_0|>`, which vLLM's phi4siglip port uses as the image
    /// placeholder id.
    static let defaultImageTokenIndex = 100_256

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mmHiddenSize = "mm_hidden_size"
        case mmProjectorType = "mm_projector_type"
        case maxNumPatches = "max_num_patches"
        case minNumPatches = "min_num_patches"
        case imageTokenIndex = "image_token_index"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType
        visionConfiguration = try c.decode(
            SharpAIPhi4SiglipVisionConfiguration.self, forKey: .visionConfiguration)

        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1.0
        let ropeScaling = try c.decodeIfPresent(JSONValue.self, forKey: .ropeScaling)
        hasRopeScaling = ropeScaling != nil
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        mmHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .mmHiddenSize)
            ?? visionConfiguration.hiddenSize
        mmProjectorType =
            try c.decodeIfPresent(String.self, forKey: .mmProjectorType) ?? "mlp2x_gelu"

        maxNumPatches = try c.decodeIfPresent(Int.self, forKey: .maxNumPatches) ?? 3600
        minNumPatches = try c.decodeIfPresent(Int.self, forKey: .minNumPatches) ?? 256
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex)
    }

    /// Placeholder id for image rows: the config key when present, otherwise the Phi-4
    /// `<|dummy_0|>` id when the vocabulary contains it.
    public var resolvedImageTokenIndex: Int? {
        if let imageTokenIndex { return imageTokenIndex }
        return vocabularySize > Self.defaultImageTokenIndex ? Self.defaultImageTokenIndex : nil
    }

    public var headDim: Int { hiddenSize / attentionHeads }

    /// Rotary dimensions, `head_dim * partial_rotary_factor`.
    public var ropeDimensions: Int { Int(Float(headDim) * partialRotaryFactor) }

    /// Side length of the square position table in the vision tower.
    public var positionGridSide: Int {
        Int(Double(visionConfiguration.positionCount).squareRoot().rounded())
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: model_type must be \(Self.expectedModelType), got \(modelType)")
        }
        guard !hasRopeScaling else {
            throw SharpAIPhi4SiglipError.unsupportedRopeScaling
        }

        let vision = visionConfiguration
        guard vision.hiddenSize > 0, vision.attentionHeads > 0,
            vision.hiddenSize % vision.attentionHeads == 0,
            vision.patchSize > 0, vision.channels > 0, vision.hiddenLayers > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: vision hidden size, heads, patch size and layers must be positive")
        }
        guard vision.hiddenAct == "gelu_pytorch_tanh" || vision.hiddenAct == "gelu" else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: unsupported vision hidden_act \(vision.hiddenAct)")
        }
        let side = positionGridSide
        guard side > 0, side * side == vision.positionCount else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: vision num_patches \(vision.positionCount) is not a square grid")
        }

        guard mmProjectorType == "mlp2x_gelu" else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: unsupported mm_projector_type \(mmProjectorType)")
        }
        guard mmHiddenSize == vision.hiddenSize else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: mm_hidden_size \(mmHiddenSize) differs from vision hidden size "
                    + "\(vision.hiddenSize)")
        }

        guard hiddenSize > 0, attentionHeads > 0, hiddenSize % attentionHeads == 0,
            kvHeads > 0, attentionHeads % kvHeads == 0, vocabularySize > 0,
            hiddenLayers > 0, intermediateSize > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: text heads, layers and vocab_size are inconsistent")
        }
        guard partialRotaryFactor > 0, partialRotaryFactor <= 1, ropeDimensions >= 2 else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: partial_rotary_factor \(partialRotaryFactor) gives no rotary dims")
        }

        guard minNumPatches > 0, maxNumPatches >= minNumPatches else {
            throw ModelFactoryError.invalidConfiguration(
                "phi4-siglip: max_num_patches must be at least min_num_patches")
        }
    }
}
