import Foundation
import MLXLMCommon

/// ops_colqwen3 config: the Qwen3-VL backbone config plus the output width `dims`.
public struct SharpAIOpsColqwen3Configuration: Decodable, Sendable, ModelConfigurationValidating {

    public let backbone: Qwen3VLConfiguration
    public let dims: Int

    enum CodingKeys: String, CodingKey {
        case dims
    }

    public init(backbone: Qwen3VLConfiguration, dims: Int) {
        self.backbone = backbone
        self.dims = dims
    }

    public init(from decoder: any Decoder) throws {
        backbone = try Qwen3VLConfiguration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        dims = try container.decode(Int.self, forKey: .dims)
    }

    public func validateModelConfiguration() throws {
        guard dims > 0 else {
            throw SharpAIOpsColqwen3Error.invalidConfiguration("dims must be positive, got \(dims)")
        }
        guard backbone.textConfiguration.hiddenSize > 0 else {
            throw SharpAIOpsColqwen3Error.invalidConfiguration("text_config.hidden_size must be positive")
        }
    }
}

public enum SharpAIOpsColqwen3Error: Error, Equatable {
    case invalidConfiguration(String)
    case featureTokenMismatch(expected: Int, actual: Int)
    case missingImageGrid
    case invalidAttentionMask(shape: [Int])
}
