// SharpAI fork overlay: glm5_next creator. Not part of upstream mlx-swift-lm.
//
// The text decoder exists (SharpAIGlm5NextTextModel, prefill forward, random-weight tested), but
// this creator stays gated: it has no KV or linear-state cache and no LanguageModel conformance,
// so it cannot generate. It decodes and validates the config, then fails with a clear error at
// load time. VLMModelFactory is not edited; the overlay consumer merges these creators into its
// own registry.
//
// Known gaps (not implemented): KV and linear-state caches, LanguageModel conformance and the
// checkpoint key mapping, the MTP drafter (Glm5NextMTPModel), the vision tower, and the chunked
// KDA kernel. KDA, NoPE MLA, the kpool DSA indexer and mHC exist in the text decoder.

import Foundation
import MLXLMCommon

public enum SharpAIGlm5NextOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "glm5_next": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIGlm5NextConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            throw SharpAIGlm5NextError.forwardNotImplemented(
                configuration.modelType ?? SharpAIGlm5NextConfiguration.expectedModelType)
        }
    ]
}
