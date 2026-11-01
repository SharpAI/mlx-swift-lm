// SharpAI fork overlay: iquestcoder creator. Not part of upstream mlx-swift-lm.
//
// iquestcoder is a Llama-identical architecture. mlx-lm maps it to "llama"
// (MODEL_REMAPPING in mlx_lm/utils.py), so this creator reuses LlamaConfiguration
// and LlamaModel. LLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIIquestcoderOverlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "iquestcoder": { data in
            try sharpAIIquestcoderCreate(data)
        }
    ]
}

/// Mirrors the private `create(LlamaConfiguration.self, LlamaModel.init)` in LLMModelFactory.
private func sharpAIIquestcoderCreate(_ data: Data) throws -> any LanguageModel {
    let configuration = try JSONDecoder.json5().decode(LlamaConfiguration.self, from: data)
    if let validating = configuration as? ModelConfigurationValidating {
        try validating.validateModelConfiguration()
    }
    return LlamaModel(configuration)
}
