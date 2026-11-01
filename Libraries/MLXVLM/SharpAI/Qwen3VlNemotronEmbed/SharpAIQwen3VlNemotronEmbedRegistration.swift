// SharpAI fork overlay: qwen3_vl_nemotron_embed creator. Not part of upstream mlx-swift-lm.
//
// The model is an embedder, not a LanguageModel, so it is not merged into VLMTypeRegistry.
// VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

/// Same decode, validate, construct sequence as the creators in `VLMModelFactory`.
private func create<C: Decodable>(
    _ configurationType: C.Type,
    _ modelInit: @escaping (C) -> SharpAIQwen3VlNemotronEmbedModel
) -> (Data) throws -> SharpAIQwen3VlNemotronEmbedModel {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return modelInit(configuration)
    }
}

public enum SharpAIQwen3VlNemotronEmbedOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators:
        [String: (Data) throws -> SharpAIQwen3VlNemotronEmbedModel] = [
            "qwen3_vl_nemotron_embed": create(
                SharpAIQwen3VlNemotronEmbedConfiguration.self,
                SharpAIQwen3VlNemotronEmbedModel.init)
        ]
}
