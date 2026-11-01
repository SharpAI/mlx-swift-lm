// SharpAI fork overlay: minicpm_ocr configuration. Not part of upstream mlx-swift-lm.
//
// Field names and defaults follow `configuration_minicpm_ocr.py` in huggingface/transformers
// (MiniCPMOcrConfig, MiniCPMOcrVisionConfig) and openbmb/MiniCPM-OCR-1.0 config.json.
// Text backbone is `qwen3_5_text`, reused from the Qwen35 text model.

import Foundation
import MLXLMCommon

public enum SharpAIMinicpmOcrError: LocalizedError {
    case unsupported(String)
    case invalidImageGrid(height: Int, width: Int, mergeFactor: Int)
    case invalidPixelShape([Int])
    case placeholderMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupported(let detail):
            return "MiniCPM-OCR unsupported: \(detail)"
        case .invalidImageGrid(let height, let width, let mergeFactor):
            return "minicpm_ocr: patch grid \(height)x\(width) must divide by \(mergeFactor)."
        case .invalidPixelShape(let shape):
            return "minicpm_ocr: pixels must be (slices, H, W, 3), got \(shape)."
        case .placeholderMismatch(let expected, let actual):
            return
                "minicpm_ocr: prompt has \(expected) image placeholders, "
                + "but the vision tower produced \(actual) tokens."
        }
    }
}

/// `vision_config` of `minicpm_ocr` (transformers `MiniCPMOcrVisionConfig`).
/// `insert_layer_id` lives at the top level and is copied into this config by the reference.
public struct SharpAIMinicpmOcrVisionConfiguration: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numChannels: Int
    public let imageSize: Int
    public let patchSize: Int
    public let hiddenAct: String
    public let layerNormEps: Float
    public let windowKernelSize: [Int]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numChannels = "num_channels"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case hiddenAct = "hidden_act"
        case layerNormEps = "layer_norm_eps"
        case windowKernelSize = "window_kernel_size"
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
        hiddenAct =
            try container.decodeIfPresent(String.self, forKey: .hiddenAct)
            ?? "gelu_pytorch_tanh"
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        windowKernelSize =
            try container.decodeIfPresent([Int].self, forKey: .windowKernelSize) ?? [2, 2]
    }
}

/// Top-level `config.json` for `model_type: minicpm_ocr`.
public struct SharpAIMinicpmOcrConfiguration: Codable, Sendable, ModelConfigurationValidating {
    public let modelType: String
    public let textConfiguration: Qwen35Configuration.TextConfiguration
    public let visionConfiguration: SharpAIMinicpmOcrVisionConfiguration
    /// Vision layer after which the window merger runs (used only when not in `4x` mode).
    public let insertLayerId: Int
    /// `"4x"` skips the window merger and keeps 4x more visual tokens. Any other value runs it.
    public let downsampleMode: String
    public let mergeKernelSize: [Int]
    public let mergerTimes: Int
    public let imageTokenId: Int
    public let videoTokenId: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case insertLayerId = "insert_layer_id"
        case downsampleMode = "downsample_mode"
        case mergeKernelSize = "merge_kernel_size"
        case mergerTimes = "merger_times"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // `model_type` is required: a config without it does not decode at all.
        modelType = try container.decode(String.self, forKey: .modelType)
        textConfiguration =
            try container.decodeIfPresent(
                Qwen35Configuration.TextConfiguration.self, forKey: .textConfiguration)
            ?? JSONDecoder().decode(
                Qwen35Configuration.TextConfiguration.self, from: Data("{}".utf8))
        visionConfiguration =
            try container.decodeIfPresent(
                SharpAIMinicpmOcrVisionConfiguration.self, forKey: .visionConfiguration)
            ?? JSONDecoder().decode(
                SharpAIMinicpmOcrVisionConfiguration.self, from: Data("{}".utf8))
        insertLayerId = try container.decodeIfPresent(Int.self, forKey: .insertLayerId) ?? 6
        downsampleMode =
            try container.decodeIfPresent(String.self, forKey: .downsampleMode) ?? "4x"
        mergeKernelSize =
            try container.decodeIfPresent([Int].self, forKey: .mergeKernelSize) ?? [2, 2]
        mergerTimes = try container.decodeIfPresent(Int.self, forKey: .mergerTimes) ?? 1
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248_056
        videoTokenId = try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248_057
    }

    /// Whether the window-attention merger runs after `insertLayerId`. Matches the reference:
    /// `use_vit_merger = downsample_mode != "4x"`.
    public var useVitMerger: Bool { downsampleMode != "4x" }

    public func validateModelConfiguration() throws {
        guard modelType == "minicpm_ocr" else {
            throw SharpAIMinicpmOcrError.unsupported("model_type \(modelType)")
        }
        guard textConfiguration.hiddenLayers > 0, textConfiguration.vocabularySize > 0 else {
            throw SharpAIMinicpmOcrError.unsupported(
                "text_config needs layers and a vocabulary")
        }
        let vision = visionConfiguration
        guard vision.numAttentionHeads > 0, vision.hiddenSize % vision.numAttentionHeads == 0
        else {
            throw SharpAIMinicpmOcrError.unsupported(
                "vision hidden_size must divide by num_attention_heads")
        }
        guard vision.patchSize > 0, vision.imageSize % vision.patchSize == 0 else {
            throw SharpAIMinicpmOcrError.unsupported(
                "vision image_size must divide by patch_size")
        }
        guard vision.hiddenAct == "gelu_pytorch_tanh" else {
            throw SharpAIMinicpmOcrError.unsupported("vision hidden_act \(vision.hiddenAct)")
        }
        guard vision.windowKernelSize == [2, 2] else {
            throw SharpAIMinicpmOcrError.unsupported(
                "window_kernel_size \(vision.windowKernelSize); only [2, 2] is implemented")
        }
        guard mergeKernelSize == [2, 2] else {
            throw SharpAIMinicpmOcrError.unsupported(
                "merge_kernel_size \(mergeKernelSize); only [2, 2] is implemented")
        }
        guard mergerTimes >= 1 else {
            throw SharpAIMinicpmOcrError.unsupported("merger_times \(mergerTimes)")
        }
        if useVitMerger {
            guard insertLayerId >= 0, insertLayerId < vision.numHiddenLayers else {
                throw SharpAIMinicpmOcrError.unsupported(
                    "insert_layer_id \(insertLayerId) outside 0..<\(vision.numHiddenLayers)")
            }
        }
    }

    /// Builds the Qwen35 configuration that the text backbone needs.
    ///
    /// Only `text_config` is used. The `vision_config` block is a placeholder that satisfies
    /// the Qwen35 decoder, and no Qwen3-VL tower is built from it.
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
