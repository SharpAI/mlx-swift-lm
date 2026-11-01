// SharpAI fork overlay: MiniCPM-V 4.7 configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow transformers `configuration_minicpmv4_7.py`
// (MiniCPMV4_7Config, MiniCPMV4_7VisionConfig) and the openbmb/MiniCPM-V-4.7-1B config.json.
// The text backbone is `qwen3_5_text`, decoded by Qwen35Configuration.TextConfiguration.
// The special token ids have no tokenizer behind them here, so they default to the values in
// the 1B config.json. Tokenizer-resolved ids in a checkpoint override them.

import Foundation
import MLXLMCommon

public enum SharpAIMinicpmv47Error: LocalizedError {
    case invalidPixelShape([Int])
    case invalidImageGrid(height: Int, width: Int, mergeFactor: Int)
    case placeholderMismatch(expected: Int, actual: Int)
    case canvasGridMismatch(String)

    public var errorDescription: String? {
        switch self {
        case .invalidPixelShape(let shape):
            return "minicpmv4_7: pixels must be (slices, H, W, 3), got \(shape)."
        case .invalidImageGrid(let height, let width, let mergeFactor):
            return
                "minicpmv4_7: patch grid \(height)x\(width) must divide by \(mergeFactor)."
        case .placeholderMismatch(let expected, let actual):
            return
                "minicpmv4_7: prompt has \(expected) placeholder positions, "
                + "but the vision tower produced \(actual) tokens."
        case .canvasGridMismatch(let detail):
            return "minicpmv4_7: canvas M-RoPE grids do not match the crops: \(detail)."
        }
    }
}

public struct SharpAIMinicpmv47VisionConfiguration: Decodable, Sendable {
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
        // Class defaults of MiniCPMV4_7VisionConfig in transformers.
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
        numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numChannels = try container.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        imageSize = try container.decodeIfPresent(Int.self, forKey: .imageSize) ?? 224
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
    }
}

public struct SharpAIMinicpmv47Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    /// Optional at decode time so that a bare `{}` still reports the model type. Required by
    /// `validateModelConfiguration()`, which runs before the model is built.
    public let textConfiguration: Qwen35Configuration.TextConfiguration?
    public let visionConfiguration: SharpAIMinicpmv47VisionConfiguration?
    /// Vision layer after which the window-attention merger runs.
    public let insertLayerId: Int
    /// `"16x"` runs the window merger and merges 4x4 patches per token. `"4x"` skips the window merger.
    public let downsampleMode: String
    public let mergerTimes: Int
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let imageStartId: Int
    public let imageEndId: Int
    public let sliceStartId: Int
    public let sliceEndId: Int
    public let newlineId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case insertLayerId = "insert_layer_id"
        case downsampleMode = "downsample_mode"
        case mergerTimes = "merger_times"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case imageStartId = "image_start_id"
        case imageEndId = "image_end_id"
        case sliceStartId = "slice_start_id"
        case sliceEndId = "slice_end_id"
        case newlineId = "newline_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "minicpmv4_7"
        textConfiguration = try container.decodeIfPresent(
            Qwen35Configuration.TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try container.decodeIfPresent(
            SharpAIMinicpmv47VisionConfiguration.self, forKey: .visionConfiguration)
        insertLayerId = try container.decodeIfPresent(Int.self, forKey: .insertLayerId) ?? 6
        downsampleMode =
            try container.decodeIfPresent(String.self, forKey: .downsampleMode) ?? "16x"
        mergerTimes = try container.decodeIfPresent(Int.self, forKey: .mergerTimes) ?? 1
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248_056
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248_057
        imageStartId = try container.decodeIfPresent(Int.self, forKey: .imageStartId) ?? 248_078
        imageEndId = try container.decodeIfPresent(Int.self, forKey: .imageEndId) ?? 248_079
        sliceStartId = try container.decodeIfPresent(Int.self, forKey: .sliceStartId) ?? 248_088
        sliceEndId = try container.decodeIfPresent(Int.self, forKey: .sliceEndId) ?? 248_089
        newlineId = try container.decodeIfPresent(Int.self, forKey: .newlineId) ?? 198
    }

    public func validateModelConfiguration() throws {
        guard !modelType.isEmpty else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: model_type must not be empty")
        }
        guard let text = textConfiguration, text.hiddenLayers > 0, text.vocabularySize > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: text_config needs layers and a vocabulary")
        }
        guard let vision = visionConfiguration else {
            throw ModelFactoryError.invalidConfiguration("minicpmv4_7: vision_config is required")
        }
        guard vision.numAttentionHeads > 0,
            vision.hiddenSize % vision.numAttentionHeads == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: vision hidden_size must divide by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: vision image_size must divide by patch_size")
        }
        guard (0 ..< vision.numHiddenLayers).contains(insertLayerId) else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: insert_layer_id must name a vision encoder layer")
        }
        guard downsampleMode == "16x" || downsampleMode == "4x" else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: downsample_mode must be \"16x\" or \"4x\", got \(downsampleMode)")
        }
        guard mergerTimes >= 1 else {
            throw ModelFactoryError.invalidConfiguration(
                "minicpmv4_7: merger_times must be at least 1")
        }
    }
}
