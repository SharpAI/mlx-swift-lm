// SharpAI fork overlay: jina_embeddings_v5 (jina-embeddings-v5-text-small) configuration.
// Not part of upstream mlx-swift-lm. The small variant is a Qwen3 decoder backbone
// (vLLM jina.py: JinaEmbeddingsV5DecoderModel). The nano variant is a different
// bidirectional encoder and is not handled here.

import Foundation
import MLXLMCommon

public struct SharpAIJinaEmbeddingsV5Configuration: Decodable, Sendable {
    /// Backbone fields, decoded by the same Qwen3 decoder used for `model_type: qwen3`.
    public let base: Qwen3Configuration
    public let taskNames: [String]
    public let jinaTask: String

    enum CodingKeys: String, CodingKey {
        case taskNames = "task_names"
        case jinaTask = "jina_task"
    }

    public init(from decoder: Decoder) throws {
        base = try Qwen3Configuration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        taskNames =
            try container.decodeIfPresent([String].self, forKey: .taskNames)
            ?? SharpAIJinaEmbeddingsV5Adapter.supportedTasks
        jinaTask =
            try container.decodeIfPresent(String.self, forKey: .jinaTask)
            ?? SharpAIJinaEmbeddingsV5Adapter.defaultTask
    }
}

extension SharpAIJinaEmbeddingsV5Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try base.validateModelConfiguration()
    }
}
