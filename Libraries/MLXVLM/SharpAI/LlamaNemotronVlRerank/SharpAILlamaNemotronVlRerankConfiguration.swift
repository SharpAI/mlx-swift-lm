// SharpAI fork overlay: llama_nemotron_vl_rerank configuration. Not part of upstream mlx-swift-lm.
//
// Checked against nvidia/llama-nemotron-rerank-vl-1b-v2 (config.json and modeling_llama_nemotron_vl.py).
// The text stack is read from `llm_config` or `text_config` when present, otherwise from the top level.
// The vision dimensions are optional at decode time, so a config without them still decodes and
// validates. Building the model needs them.

import Foundation
import MLXLMCommon

public enum SharpAILlamaNemotronVlRerankError: LocalizedError {
    case imageTokenMismatch(expected: Int, actual: Int)
    case missingImageTokenId
    case invalidPixelShape(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .imageTokenMismatch(let expected, let actual):
            return
                "llama_nemotron_vl_rerank: expected \(expected) image feature rows, "
                + "but the input has \(actual) image tokens"
        case .missingImageTokenId:
            return "llama_nemotron_vl_rerank: img_context_token_id is required to merge images"
        case .invalidPixelShape(let expected, let actual):
            return
                "llama_nemotron_vl_rerank: pixel height and width must be \(expected), got \(actual)"
        }
    }
}

public struct SharpAILlamaNemotronVlRerankConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    public let modelType: String
    public let architectures: [String]
    public let textConfiguration: SharpAILlamaNemotronVlConfiguration.TextConfiguration
    /// Nil when the config has no SigLIP dimensions. The model build needs them.
    public let visionConfiguration: SharpAILlamaNemotronVlConfiguration.VisionConfiguration?
    public let hasVisionConfig: Bool
    /// Pixel-shuffle ratio. The reference uses `downsample_ratio` 0.5, so the factor is 2.
    public let downsampleRatio: Float
    /// Pooling over tokens before the score head. The reference uses "avg", the only one implemented.
    public let pooling: String
    /// Divides the score logits. The reference default is 1.0.
    public let temperature: Float
    /// Token id that marks image rows in `input_ids` (`img_context_token_id`).
    public let imageTokenId: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case architectures
        case llmConfig = "llm_config"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case downsampleRatio = "downsample_ratio"
        case pooling
        case temperature
        case imageTokenId = "img_context_token_id"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try container.decode(String.self, forKey: .modelType)
        architectures = try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        hasVisionConfig = container.contains(.visionConfig)

        if let nestedKey = [CodingKeys.llmConfig, CodingKeys.textConfig].first(where: {
            container.contains($0)
        }) {
            textConfiguration = try container.decode(
                SharpAILlamaNemotronVlConfiguration.TextConfiguration.self, forKey: nestedKey)
        } else {
            textConfiguration = try SharpAILlamaNemotronVlConfiguration.TextConfiguration(
                from: decoder)
        }

        visionConfiguration = try? container.decodeIfPresent(
            SharpAILlamaNemotronVlConfiguration.VisionConfiguration.self, forKey: .visionConfig)
        downsampleRatio = try container.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        pooling = try container.decodeIfPresent(String.self, forKey: .pooling) ?? "avg"
        temperature = try container.decodeIfPresent(Float.self, forKey: .temperature) ?? 1.0
        imageTokenId = try container.decodeIfPresent(Int.self, forKey: .imageTokenId)
    }

    /// Integer `k` with `downsample_ratio == 1 / k`. Returns 0 when no valid `k` exists.
    public var pixelShuffleFactor: Int {
        guard downsampleRatio.isFinite, downsampleRatio > 0 else { return 0 }
        let k = (1 / downsampleRatio).rounded()
        guard k < 1_000_000 else { return 0 }
        return Int(k)
    }

    public func validateModelConfiguration() throws {
        let text = textConfiguration
        let kvHeads = text.kvHeads
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.attentionHeads > 0, kvHeads > 0,
            text.vocabularySize > 0, text.headDim > 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: text hidden, head and vocab sizes must be positive")
        }
        guard text.attentionHeads % kvHeads == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: num_attention_heads must be divisible by num_key_value_heads")
        }
        guard pooling == "avg" else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: pooling \(pooling) is not implemented, only \"avg\"")
        }
        guard temperature.isFinite, temperature > 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: temperature must be a positive number")
        }

        let factor = pixelShuffleFactor
        guard factor >= 1, abs(1 / Float(factor) - downsampleRatio) < 1e-6 else {
            throw ModelFactoryError.invalidConfiguration(
                "llama_nemotron_vl_rerank: downsample_ratio must be 1/k for a whole number k, got \(downsampleRatio)")
        }

        if let vision = visionConfiguration {
            guard vision.hiddenSize > 0, vision.hiddenLayers > 0, vision.intermediateSize > 0,
                vision.attentionHeads > 0, vision.patchSize > 0, vision.imageSize > 0
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "llama_nemotron_vl_rerank: vision sizes must be positive")
            }
            guard vision.hiddenSize % vision.attentionHeads == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "llama_nemotron_vl_rerank: vision hidden_size must be divisible by num_attention_heads")
            }
            guard vision.imageSize % vision.patchSize == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "llama_nemotron_vl_rerank: vision image_size must be divisible by patch_size")
            }
            let grid = vision.imageSize / vision.patchSize
            guard grid % factor == 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "llama_nemotron_vl_rerank: vision grid (\(grid)) must be divisible by the pixel-shuffle factor (\(factor))")
            }
        }
    }
}
