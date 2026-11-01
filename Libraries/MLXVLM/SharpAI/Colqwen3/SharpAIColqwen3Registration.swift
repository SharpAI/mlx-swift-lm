// SharpAI fork overlay: colqwen3 creator. Not part of upstream mlx-swift-lm.
//
// Returns the multi-vector encoder, not a LanguageModel, so it is not merged into
// VLMTypeRegistry. VLMModelFactory is not edited.

import Foundation
import MLXLMCommon

/// Same decode, validate, construct sequence as the creators in `VLMModelFactory`.
private func create<C: Decodable>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> SharpAIColqwen3Model
) -> (Data) throws -> SharpAIColqwen3Model {
    { data in
        let configuration = try JSONDecoder.json5().decode(C.self, from: data)
        if let validating = configuration as? ModelConfigurationValidating {
            try validating.validateModelConfiguration()
        }
        return modelInit(configuration)
    }
}

public enum SharpAIColqwen3Overlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> SharpAIColqwen3Model] = [
        "colqwen3": create(SharpAIColqwen3Configuration.self, SharpAIColqwen3Model.init)
    ]
}
