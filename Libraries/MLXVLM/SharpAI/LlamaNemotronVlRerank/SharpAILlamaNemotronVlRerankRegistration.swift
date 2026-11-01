// SharpAI fork overlay: llama_nemotron_vl_rerank creator. Not part of upstream mlx-swift-lm.
//
// Decodes and validates the config, then builds SharpAILlamaNemotronVlRerankModel. VLMTypeRegistry
// and the reranker factory are not edited. The reranker factory has no VLM architecture case, so
// this overlay is not reachable from RerankerModelFactory yet.

import Foundation
import MLXLMCommon

public enum SharpAILlamaNemotronVlRerankOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "llama_nemotron_vl_rerank": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAILlamaNemotronVlRerankConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAILlamaNemotronVlRerankModel(configuration)
        }
    ]
}
