// SharpAI fork overlay: kimi_k3 creator. Not part of upstream mlx-swift-lm.
//
// Creators use the same closure shape as VLMTypeRegistry: decode, validate, construct.
// VLMModelFactory is not edited. A consumer merges these creators into its registry.
//
// The construct step builds the text backbone (SharpAIKimiK3Model) from `text_config`. The vision tower and
// mm_projector are NOT ported, so the model is text-only: `vision_config` is validated by decode but unused, and
// image inputs cannot reach the model. Checkpoints with MXFP4-packed experts still need the expert mapping
// (see SharpAIKimiK3Sanitizer) before they load.

import Foundation
import MLXLMCommon

public enum SharpAIKimiK3Overlay {

    /// Creators keyed by `model_type`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "kimi_k3": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAIKimiK3Configuration.self, from: data)
            try configuration.validateModelConfiguration()
            let text = try configuration.decodedTextConfiguration()
            try text.validateForConstruction()
            return SharpAIKimiK3Model(text)
        }
    ]
}
