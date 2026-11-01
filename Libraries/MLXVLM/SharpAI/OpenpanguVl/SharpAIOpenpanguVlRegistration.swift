// SharpAI fork overlay: openpangu_vl creator. Not part of upstream mlx-swift-lm.
//
// The creator decodes and validates the config, then builds the text decoder
// (SharpAIOpenpanguVlModel). The ViT is not implemented, so image and video input throws.
// VLMModelFactory is not edited; the overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIOpenpanguVlOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "openpangu_vl": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIOpenpanguVlConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIOpenpanguVlModel(configuration)
        }
    ]
}
