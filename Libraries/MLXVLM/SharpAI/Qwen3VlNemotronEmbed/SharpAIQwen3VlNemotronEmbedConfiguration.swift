import Foundation
import MLXLMCommon

/// The `pooling` value read from config. Verified against: https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/resolve/main/config.json
/// (`"pooling": "colbert"`). `colbert` means per-token output, which is what `embed` returns. The reference
/// modeling file stores `pooling` but never reads it, so `embed` ignores this setting.
/// UNVERIFIED: `last` and `mean` are local names kept for the pooling helper. No checkpoint or reference uses them.
public enum SharpAIQwen3VlNemotronEmbedPooling: String, Codable, Sendable {
    case colbert
    case last
    case mean
}

/// Qwen3-VL backbone config plus the embedding head settings.
///
/// `embedDim` is not a checkpoint key. The reference checkpoint config has no `embed_dim`, and its output width is the
/// text hidden size. `embed` always returns the text hidden size, so this field does not change the output.
/// UNVERIFIED: the `embed_dim` key is a local leftover, kept so existing decoding tests still pass.
public struct SharpAIQwen3VlNemotronEmbedConfiguration: Decodable, Sendable, ModelConfigurationValidating {

    public let backbone: Qwen3VLConfiguration
    public let embedDim: Int
    public let pooling: SharpAIQwen3VlNemotronEmbedPooling

    enum CodingKeys: String, CodingKey {
        case embedDim = "embed_dim"
        case pooling
    }

    public init(
        backbone: Qwen3VLConfiguration, embedDim: Int? = nil,
        pooling: SharpAIQwen3VlNemotronEmbedPooling = .colbert
    ) {
        self.backbone = backbone
        self.embedDim = embedDim ?? backbone.textConfiguration.hiddenSize
        self.pooling = pooling
    }

    public init(from decoder: any Decoder) throws {
        let backbone = try Qwen3VLConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let embedDim =
            try container.decodeIfPresent(Int.self, forKey: .embedDim)
            ?? backbone.textConfiguration.hiddenSize
        let pooling =
            try container.decodeIfPresent(
                SharpAIQwen3VlNemotronEmbedPooling.self, forKey: .pooling) ?? .colbert

        self.backbone = backbone
        self.embedDim = embedDim
        self.pooling = pooling
    }

    public func validateModelConfiguration() throws {
        guard embedDim > 0 else {
            throw SharpAIQwen3VlNemotronEmbedError.invalidConfiguration(
                "embed_dim must be positive, got \(embedDim)")
        }
        guard backbone.textConfiguration.hiddenSize > 0 else {
            throw SharpAIQwen3VlNemotronEmbedError.invalidConfiguration(
                "text_config.hidden_size must be positive")
        }
    }
}

public enum SharpAIQwen3VlNemotronEmbedError: Error, Equatable {
    case invalidConfiguration(String)
    case featureTokenMismatch(expected: Int, actual: Int)
    case missingImageGrid
    case invalidAttentionMask(shape: [Int])
    case unsupportedBatchWithImages(batch: Int)
}
