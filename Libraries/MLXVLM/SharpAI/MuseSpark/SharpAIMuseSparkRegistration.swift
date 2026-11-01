// SharpAI fork overlay: muse_spark creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.
// The construct step throws: the text backbone (SharpAIMuseSparkTextForCausalLM) exists, but the
// vision and audio towers and the checkpoint key mapping are not written, so the factory stays gated.

import Foundation
import MLXLMCommon

public enum SharpAIMuseSparkOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "muse_spark": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMuseSparkConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIMuseSparkError.unsupported(
                "vision/audio towers and checkpoint key mapping are not written; text backbone only")
        }
    ]
}
