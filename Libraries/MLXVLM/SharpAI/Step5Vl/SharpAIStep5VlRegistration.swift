// SharpAI fork overlay: step5_vl creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.
// The creator builds SharpAIStep5VlModel, the dense text decoder. It throws `unsupported` for the
// parts not implemented yet: the CSA sparse indexer, MoE layers, and the perception_encoder vision
// tower (see SharpAIStep5VlSupport in SharpAIStep5VlModel.swift).

import Foundation
import MLXLMCommon

public enum SharpAIStep5VlOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "step5_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIStep5VlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIStep5VlModel(configuration)
        }
    ]
}
