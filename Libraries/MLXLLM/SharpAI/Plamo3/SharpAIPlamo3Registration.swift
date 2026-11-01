// SharpAI fork overlay: plamo3 creator. Not part of upstream mlx-swift-lm.
//
// Same closure shape as LLMTypeRegistry: decode, validate, construct. LLMModelFactory is not
// edited; the overlay consumer merges these creators. The creator builds SharpAIPlamo3Model.
// Still missing: the custom plamo3 tokenizer (the forward pass does not need it), and the
// checkpoint key names, which are UNVERIFIED (gated repo, no config.json here).

import Foundation
import MLXLMCommon

public enum SharpAIPlamo3Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "plamo3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIPlamo3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return try SharpAIPlamo3Model(configuration)
        }
    ]
}
