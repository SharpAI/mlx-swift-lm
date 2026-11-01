// SharpAI fork overlay: diffusion_gemma configuration. Not part of upstream mlx-swift-lm.
//
// Decodes the text dimensions only. Block-diffusion fields (mask token, block length, denoise
// steps) are not decoded: no reference source for them is in the repo.
// The nested `text_config` block and the `vision_config` presence check are verified against the
// real config: Verified against: https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/config.json
// UNVERIFIED: the flat top-level fallback is not in that config, and the reference
// configuration_diffusion_gemma.py has no flat form, so it is kept only as a leniency.

import Foundation
import MLXLMCommon

public enum SharpAIDiffusionGemmaError: LocalizedError {
    case forwardNotImplemented(String)

    public var errorDescription: String? {
        switch self {
        case .forwardNotImplemented(let type):
            return
                "Model type \(type) is registered, but its forward pass and denoise engine "
                + "are not implemented yet."
        }
    }
}

/// Text dimensions for diffusion_gemma. All fields are required: no defaults are invented.
public struct SharpAIDiffusionGemmaTextDimensions: Decodable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
    }
}

public struct SharpAIDiffusionGemmaConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let textDimensions: SharpAIDiffusionGemmaTextDimensions
    public let hasVisionConfig: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? "diffusion_gemma"

        if container.contains(.textConfig) {
            textDimensions = try container.decode(
                SharpAIDiffusionGemmaTextDimensions.self, forKey: .textConfig)
        } else {
            textDimensions = try SharpAIDiffusionGemmaTextDimensions(from: decoder)
        }

        hasVisionConfig = container.contains(.visionConfig)
    }

    public func validateModelConfiguration() throws {
        let text = textDimensions
        let dims = [
            text.hiddenSize, text.numHiddenLayers, text.numAttentionHeads,
            text.numKeyValueHeads, text.vocabSize,
        ]
        guard dims.allSatisfy({ $0 > 0 }) else {
            throw ModelFactoryError.invalidConfiguration(
                "diffusion_gemma: text hidden, layer, head and vocab sizes must be positive")
        }
        guard text.numAttentionHeads % text.numKeyValueHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "diffusion_gemma: num_attention_heads must be divisible by num_key_value_heads")
        }
    }
}
