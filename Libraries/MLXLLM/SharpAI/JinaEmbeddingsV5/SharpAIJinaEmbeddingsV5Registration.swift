// SharpAI fork overlay: jina_embeddings_v5 creator. Not part of upstream mlx-swift-lm.
// Same decode, validate, construct sequence as the LLMTypeRegistry creators.
// LLMModelFactory is not edited. The overlay consumer merges these creators.

import Foundation
import MLXLMCommon

public enum SharpAIJinaEmbeddingsV5Overlay {

    /// Creators keyed by `model_type`, same closure shape as `LLMTypeRegistry`.
    /// `is_decoder: false` is the nano bidirectional EuroBERT encoder, which is refused here
    /// (see ``SharpAIJinaEmbeddingsV5Error/nanoVariantNotWired``). Anything else (absent or
    /// true) selects the small Qwen3 decoder backbone.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "jina_embeddings_v5": { data in
            let variant = try JSONDecoder.json5().decode(
                SharpAIJinaEmbeddingsV5Variant.self, from: data)
            if variant.isDecoder == false {
                throw SharpAIJinaEmbeddingsV5Error.nanoVariantNotWired
            }
            let configuration = try JSONDecoder.json5().decode(
                SharpAIJinaEmbeddingsV5Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            return SharpAIJinaEmbeddingsV5Model(configuration)
        }
    ]
}

/// Reads only `is_decoder`, to choose between the nano encoder and the small decoder.
struct SharpAIJinaEmbeddingsV5Variant: Decodable {
    let isDecoder: Bool?

    enum CodingKeys: String, CodingKey {
        case isDecoder = "is_decoder"
    }
}
