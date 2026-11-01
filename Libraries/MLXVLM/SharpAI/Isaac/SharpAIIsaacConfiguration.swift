// SharpAI fork overlay: isaac configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow the Perceptron/Isaac-0.1 config.json (model_type "isaac",
// vision_config model_type "isaac_vision", text_config model_type "qwen3").
// Vision is a fixed square grid only; NaFlex (variable resolution, vision_max_num_patches)
// is not implemented, so num_patches must equal grid².

import Foundation
import MLXLMCommon

public enum SharpAIIsaacError: LocalizedError {
    case unsupportedActivation(String)
    case duplicateWeightKey(String)
    case visionNotImplemented
    case emptyPrompt

    public var errorDescription: String? {
        switch self {
        case .unsupportedActivation(let name):
            return "isaac: unsupported vision hidden_act \(name)"
        case .duplicateWeightKey(let key):
            return "isaac: duplicate weight key after sanitize: \(key)"
        case .visionNotImplemented:
            return "isaac: image and video input is not implemented; splicing vision features into the token stream is missing"
        case .emptyPrompt:
            return "isaac: prompt has no tokens"
        }
    }
}

/// Siglip2-style vision tower settings (`vision_config`).
public struct SharpAIIsaacVisionConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let channels: Int
    public let patchSize: Int
    public let imageSize: Int
    public let numPatches: Int
    public let layerNormEps: Float
    public let hiddenAct: String
    /// Pixel-shuffle factor applied after the tower, before the projector (2 in Isaac-0.1).
    public let pixelShuffleScaleFactor: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case channels = "num_channels"
        case patchSize = "patch_size"
        case imageSize = "image_size"
        case numPatches = "num_patches"
        case layerNormEps = "layer_norm_eps"
        case hiddenAct = "hidden_act"
        case pixelShuffleScaleFactor = "pixel_shuffle_scale_factor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        let patch = try c.decode(Int.self, forKey: .patchSize)
        let size = try c.decode(Int.self, forKey: .imageSize)
        patchSize = patch
        imageSize = size
        let grid = size / max(patch, 1)
        numPatches = try c.decodeIfPresent(Int.self, forKey: .numPatches) ?? grid * grid
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
        pixelShuffleScaleFactor =
            try c.decodeIfPresent(Int.self, forKey: .pixelShuffleScaleFactor) ?? 2
    }

    /// Patches per side in the fixed grid.
    public var gridSide: Int { imageSize / patchSize }

    /// Length of one flattened patch: `channels * patch_size * patch_size`.
    public var patchInputSize: Int { channels * patchSize * patchSize }
}

/// Qwen3-style text decoder settings (`text_config`).
public struct SharpAIIsaacTextConfiguration: Decodable, Sendable {
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let vocabularySize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4 * hiddenSize
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 1
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / max(attentionHeads, 1)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        if let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = theta
        } else if let parameters = try c.decodeIfPresent(
            SharpAIIsaacRopeParameters.self, forKey: .ropeParameters),
            let theta = parameters.ropeTheta
        {
            ropeTheta = theta
        } else {
            ropeTheta = 1_000_000
        }
    }
}

/// `rope_parameters` in the newer transformers layout; only `rope_theta` is read.
private struct SharpAIIsaacRopeParameters: Decodable {
    let ropeTheta: Float?

    enum CodingKeys: String, CodingKey {
        case ropeTheta = "rope_theta"
    }
}

/// Top-level isaac configuration.
public struct SharpAIIsaacConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    public static let expectedModelType = "isaac"

    public let modelType: String
    public let visionConfiguration: SharpAIIsaacVisionConfiguration
    public let textConfiguration: SharpAIIsaacTextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? Self.expectedModelType
        visionConfiguration = try c.decode(
            SharpAIIsaacVisionConfiguration.self, forKey: .visionConfiguration)
        textConfiguration = try c.decode(
            SharpAIIsaacTextConfiguration.self, forKey: .textConfiguration)
    }

    public func validateModelConfiguration() throws {
        guard modelType == Self.expectedModelType else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: model_type must be \(Self.expectedModelType), got \(modelType)")
        }

        let vision = visionConfiguration
        guard vision.hiddenSize > 0, vision.attentionHeads > 0,
            vision.hiddenSize % vision.attentionHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: vision hidden_size must divide by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: vision image_size must be a multiple of patch_size")
        }
        guard vision.numPatches == vision.gridSide * vision.gridSide else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: vision num_patches \(vision.numPatches) must equal grid² "
                    + "\(vision.gridSide * vision.gridSide); NaFlex is not implemented")
        }
        guard vision.pixelShuffleScaleFactor > 0,
            vision.gridSide % vision.pixelShuffleScaleFactor == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: vision grid side must be a multiple of pixel_shuffle_scale_factor")
        }
        guard vision.hiddenAct == "gelu_pytorch_tanh" || vision.hiddenAct == "gelu" else {
            throw SharpAIIsaacError.unsupportedActivation(vision.hiddenAct)
        }

        let text = textConfiguration
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.vocabularySize > 0,
            text.intermediateSize > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: text hidden_size, num_hidden_layers, vocab_size and intermediate_size "
                    + "must be positive")
        }
        guard text.attentionHeads > 0, text.kvHeads > 0,
            text.attentionHeads % text.kvHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: text num_attention_heads must be a positive multiple of num_key_value_heads")
        }
        guard text.headDim > 0, text.headDim % 2 == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "isaac: text head_dim must be a positive even number for RoPE")
        }
    }
}
