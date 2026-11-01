// SharpAI fork overlay: MiniCPM-V 4.6 configuration. Not part of upstream mlx-swift-lm.
//
// Text backbone is `qwen3_5_text`, reused from the Qwen35 text model.
// Vision is a SigLIP-style tower with NaViT position buckets, a window merger after
// `insert_layer_id`, and an MLP merger. Field names follow openbmb/MiniCPM-V-4_6 config.json.
// UNVERIFIED: values come from the fixture in scripts/conformance, not a live download.

import Foundation
import MLXLMCommon

public enum SharpAIMinicpmv46Error: LocalizedError {
    case invalidImageGrid(height: Int, width: Int, mergeFactor: Int)
    case invalidPixelShape([Int])
    case placeholderMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .invalidImageGrid(let height, let width, let mergeFactor):
            return
                "minicpmv4_6: patch grid \(height)x\(width) must divide by \(mergeFactor)."
        case .invalidPixelShape(let shape):
            return "minicpmv4_6: pixels must be (slices, H, W, 3), got \(shape)."
        case .placeholderMismatch(let expected, let actual):
            return
                "minicpmv4_6: prompt has \(expected) image placeholders, "
                + "but the vision tower produced \(actual) tokens."
        }
    }
}

public struct SharpAIMinicpmv46VisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numChannels: Int
    public let imageSize: Int
    public let patchSize: Int
    public let layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case layerNormEps = "layer_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
        numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 27
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

public struct SharpAIMinicpmv46Configuration: Codable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let textConfiguration: Qwen35Configuration.TextConfiguration
    public let visionConfiguration: SharpAIMinicpmv46VisionConfiguration
    /// Vision layer after which the window merger runs. Negative disables the merger.
    public let insertLayerId: Int
    public let dropVisionLastLayer: Bool
    public let imageTokenId: Int
    public let videoTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case insertLayerId = "insert_layer_id"
        case dropVisionLastLayer = "drop_vision_last_layer"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "minicpmv4_6"
        textConfiguration = try container.decode(
            Qwen35Configuration.TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try container.decode(
            SharpAIMinicpmv46VisionConfiguration.self, forKey: .visionConfiguration)
        insertLayerId = try container.decodeIfPresent(Int.self, forKey: .insertLayerId) ?? -1
        dropVisionLastLayer =
            try container.decodeIfPresent(Bool.self, forKey: .dropVisionLastLayer) ?? false
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248_056
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248_057
    }

    public func validateModelConfiguration() throws {
        guard textConfiguration.hiddenLayers > 0, textConfiguration.vocabularySize > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_6: text_config needs layers and a vocabulary")
        }
        guard visionConfiguration.numAttentionHeads > 0,
            visionConfiguration.hiddenSize % visionConfiguration.numAttentionHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_6: vision hidden_size must divide by num_attention_heads")
        }
        guard visionConfiguration.patchSize > 0,
            visionConfiguration.imageSize % visionConfiguration.patchSize == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_6: vision image_size must divide by patch_size")
        }
    }

    /// Builds the Qwen35 configuration that the text backbone needs.
    ///
    /// Only `text_config` is used. The `vision_config` block is a placeholder that
    /// satisfies the Qwen35 decoder, and no Qwen3-VL tower is built from it.
    /// UNVERIFIED: depends on `Qwen35Configuration` decoding these placeholder keys.
    func makeQwen35Configuration() throws -> Qwen35Configuration {
        let textData = try JSONEncoder().encode(textConfiguration)
        let textObject: Any = try JSONSerialization.jsonObject(with: textData)
        let root: [String: Any] = [
            "model_type": "qwen3_5",
            "text_config": textObject,
            "vision_config": [
                "model_type": "qwen3_vl",
                "depth": visionConfiguration.numHiddenLayers,
                "hidden_size": visionConfiguration.hiddenSize,
                "intermediate_size": visionConfiguration.intermediateSize,
                "out_hidden_size": textConfiguration.hiddenSize,
                "num_heads": visionConfiguration.numAttentionHeads,
                "patch_size": visionConfiguration.patchSize,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "num_position_embeddings": 4096,
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: root)
        return try JSONDecoder().decode(Qwen35Configuration.self, from: data)
    }
}
