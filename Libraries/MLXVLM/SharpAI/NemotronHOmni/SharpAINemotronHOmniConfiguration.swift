// SharpAI fork overlay: nemotron_h_omni configuration. Not part of upstream mlx-swift-lm.
//
// The text stack is nested under `llm_config` and decoded with NemotronHConfiguration, which
// already reads `layers_block_type` and `moe_latent_size`. Vision and sound configs stay raw.
// Verified against: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/config.json
// Confirmed top-level keys: model_type, architectures, llm_config, vision_config, sound_config,
// projector_hidden_size, downsample_ratio, patch_size, force_image_size. The `vit_hidden_size` key
// is not confirmed at top level; visionHiddenSize stays as a best-effort decode.

import Foundation
import MLXLLM
import MLXLMCommon

public struct SharpAINemotronHOmniConfiguration: Codable, Sendable {
    public var modelType: String
    public var architectures: [String]
    public var llmConfig: NemotronHConfiguration
    public var visionConfig: JSONValue?
    public var soundConfig: JSONValue?
    public var projectorHiddenSize: Int?
    public var downsampleRatio: Double?
    public var patchSize: Int?
    public var forceImageSize: Int?
    public var videoTemporalPatchSize: Int?
    public var visionHiddenSize: Int?

    public static let defaultModelType = "nemotron_h_omni"

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case llmConfig = "llm_config"
        case visionConfig = "vision_config"
        case soundConfig = "sound_config"
        case projectorHiddenSize = "projector_hidden_size"
        case downsampleRatio = "downsample_ratio"
        case patchSize = "patch_size"
        case forceImageSize = "force_image_size"
        case videoTemporalPatchSize = "video_temporal_patch_size"
        case visionHiddenSize = "vit_hidden_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? Self.defaultModelType
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        llmConfig = try container.decode(NemotronHConfiguration.self, forKey: .llmConfig)
        visionConfig = try container.decodeIfPresent(JSONValue.self, forKey: .visionConfig)
        soundConfig = try container.decodeIfPresent(JSONValue.self, forKey: .soundConfig)
        projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize)
        downsampleRatio = try container.decodeIfPresent(Double.self, forKey: .downsampleRatio)
        patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize)
        forceImageSize = try container.decodeIfPresent(Int.self, forKey: .forceImageSize)
        videoTemporalPatchSize =
            try container.decodeIfPresent(Int.self, forKey: .videoTemporalPatchSize)
        visionHiddenSize = try container.decodeIfPresent(Int.self, forKey: .visionHiddenSize)
    }
}

extension SharpAINemotronHOmniConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard llmConfig.vocabSize > 0, llmConfig.hiddenSize > 0 else {
            throw SharpAINemotronHOmniError.invalidConfiguration(
                "llm_config needs a positive vocab_size and hidden_size")
        }
        guard !llmConfig.hybridOverridePattern.isEmpty else {
            throw SharpAINemotronHOmniError.invalidConfiguration(
                "llm_config needs a non-empty layer list")
        }
        if let latent = llmConfig.moeLatentSize, latent <= 0 {
            throw SharpAINemotronHOmniError.invalidConfiguration(
                "llm_config.moe_latent_size must be positive")
        }
        if let ratio = downsampleRatio, ratio <= 0 {
            throw SharpAINemotronHOmniError.invalidConfiguration(
                "downsample_ratio must be positive")
        }
    }
}

public enum SharpAINemotronHOmniError: LocalizedError, Equatable {
    case invalidConfiguration(String)
    case duplicateWeightKey(String)
    case visionTowerNotImplemented

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let detail):
            return "Invalid nemotron_h_omni configuration: \(detail)"
        case .duplicateWeightKey(let key):
            return "Duplicate weight key after nemotron_h_omni sanitize: \(key)"
        case .visionTowerNotImplemented:
            return "nemotron_h_omni image, video and audio input need the vision and sound towers (not implemented)."
        }
    }
}
