import Foundation
import MLXLMCommon

/// ColQwen3 config: the Qwen3-VL backbone config plus the projection size `embed_dim`.
public struct SharpAIColqwen3Configuration: Decodable, Sendable, ModelConfigurationValidating {

    public let backbone: Qwen3VLConfiguration
    public let embedDim: Int

    enum CodingKeys: String, CodingKey {
        case embedDim = "embed_dim"
    }

    public init(backbone: Qwen3VLConfiguration, embedDim: Int) {
        self.backbone = backbone
        self.embedDim = embedDim
    }

    public init(from decoder: any Decoder) throws {
        backbone = try Qwen3VLConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        embedDim = try container.decode(Int.self, forKey: .embedDim)
    }

    public func validateModelConfiguration() throws {
        guard embedDim > 0 else {
            throw SharpAIColqwen3Error.invalidConfiguration("embed_dim must be positive, got \(embedDim)")
        }
        guard backbone.textConfiguration.hiddenSize > 0 else {
            throw SharpAIColqwen3Error.invalidConfiguration("text_config.hidden_size must be positive")
        }
    }
}

public enum SharpAIColqwen3Error: Error, Equatable {
    case invalidConfiguration(String)
    case featureTokenMismatch(expected: Int, actual: Int)
    case missingImageGrid
    case invalidAttentionMask(shape: [Int])
}
