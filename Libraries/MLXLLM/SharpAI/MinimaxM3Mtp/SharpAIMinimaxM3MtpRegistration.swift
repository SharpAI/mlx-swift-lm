// SharpAI fork overlay: minimax_m3_mtp creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as LLMTypeRegistry: decode, validate, construct.
// LLMModelFactory is not edited. A consumer merges these creators into its registry.
//
// The MTP head itself is implemented in SharpAIMinimaxM3MtpModel.swift (SharpAIMinimaxM3MtpHead).
// The creator still throws after validation. It cannot return a `LanguageModel`, because the head
// has no standalone forward: it needs the target model's token embeddings, hidden states and
// lm_head, and the target-side speculative path that would supply them is not in the repo.
// The public M3 checkpoint also has no `mtp.*` tensors to load.

import Foundation
import MLXLMCommon

public enum SharpAIMinimaxM3MtpOverlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "minimax_m3_mtp": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIMinimaxM3MtpConfiguration.self, from: data)
            try configuration.validateModelConfiguration()

            let text: SharpAIMinimaxM3TextConfiguration
            let depth: SharpAIMinimaxM3MtpDepthConfiguration
            do {
                text = try JSONDecoder.json5().decode(
                    SharpAIMinimaxM3TextConfiguration.self, from: data)
                depth = try JSONDecoder.json5().decode(
                    SharpAIMinimaxM3MtpDepthConfiguration.self, from: data)
            } catch {
                throw SharpAIMinimaxM3MtpError.invalidConfiguration(
                    "text fields missing or malformed: \(error)")
            }
            try SharpAIMinimaxM3MtpHead.validate(text, numMtpModules: depth.numMtpModules)

            throw SharpAIMinimaxM3MtpError.unsupported(
                "the MTP head builds (SharpAIMinimaxM3MtpHead) but has no standalone LanguageModel "
                    + "forward: it needs the target's embeddings, hidden states and lm_head")
        }
    ]
}
