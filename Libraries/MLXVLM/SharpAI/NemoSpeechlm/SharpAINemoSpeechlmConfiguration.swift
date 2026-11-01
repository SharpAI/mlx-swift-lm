// SharpAI fork overlay: nemo_speechlm configuration. Not part of upstream mlx-swift-lm.
//
// Verified against: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/resolve/main/config.json
// (model_type, architectures, and the top-level blocks `perception` and `llm_config`).
// Only the top-level keys are decoded here. The encoder hyperparameters are decoded by
// SharpAINemoSpeechlmEncoderConfiguration from the `perception.encoder` block.

import Foundation
import MLXLMCommon

public enum SharpAINemoSpeechlmError: LocalizedError {
    case forwardNotImplemented(String)
    case emptyWeightKey
    case unmappedDecoderWeight(String)
    case unsupportedEncoder(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass is not implemented yet "
                + "(audio speaker branch and LLM embedding entry point missing)."
        case .emptyWeightKey:
            return "Empty weight key in nemo_speechlm checkpoint."
        case .unmappedDecoderWeight(let key):
            return "nemo_speechlm checkpoint key has no decoder mapping: \(key)"
        case .unsupportedEncoder(let reason):
            return "nemo_speechlm audio encoder configuration is not supported: \(reason)"
        }
    }
}

public struct SharpAINemoSpeechlmConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    /// Verified against: the `architectures` field of
    /// https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/resolve/main/config.json
    /// (`Nemotron3_5TranscribeForConditionalGeneration`). The MTP variant name is not in that config.
    /// UNVERIFIED: `Nemotron3_5TranscribeMTPModel` has no source in this repo. Remove it if no source names it.
    public static let knownArchitectures: Set<String> = [
        "Nemotron3_5TranscribeForConditionalGeneration",
        "Nemotron3_5TranscribeMTPModel",
    ]

    public let modelType: String
    public let architectures: [String]
    /// Fixture-only keys (`audio_config`, `text_config`) kept for the existing tests. The real checkpoint
    /// names its sub-configs `perception` and `llm_config`, decoded below as `hasPerception` and `hasLLMConfig`.
    public let hasAudioConfig: Bool
    public let hasTextConfig: Bool
    /// The real checkpoint's top-level `perception` block (audio tower hyperparameters).
    public let hasPerception: Bool
    /// The real checkpoint's top-level `llm_config` block (NemotronH hyperparameters).
    public let hasLLMConfig: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case perception
        case llmConfig = "llm_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "nemo_speechlm"
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hasAudioConfig = container.contains(.audioConfig)
        hasTextConfig = container.contains(.textConfig)
        hasPerception = container.contains(.perception)
        hasLLMConfig = container.contains(.llmConfig)
    }

    public func validateModelConfiguration() throws {
        guard !modelType.isEmpty else {
            throw ModelFactoryError.invalidConfiguration(
                "nemo_speechlm: model_type must not be empty")
        }
        let knownMatch = architectures.contains { Self.knownArchitectures.contains($0) }
        guard architectures.isEmpty || knownMatch else {
            throw ModelFactoryError.invalidConfiguration(
                "nemo_speechlm: architectures \(architectures) are not a known Transcribe architecture")
        }
    }
}
