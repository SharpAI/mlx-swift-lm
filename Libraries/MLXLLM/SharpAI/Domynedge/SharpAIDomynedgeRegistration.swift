// SharpAI fork overlay: domynedge creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape and decode, validate, construct order as LLMTypeRegistry.
// LLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIDomynedgeOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "domynedge": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIDomynedgeConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIDomynedgeModel(configuration)
        }
    ]
}
