// SharpAI fork overlay: cosmos3_omni creator. Not part of upstream mlx-swift-lm.
//
// The understanding path is Qwen3-VL. In the transformers reference, `Cosmos3OmniModel` builds
// `visual` from `vision_config` and `language_model` from `text_config`, both through AutoModel
// (Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_omni/modeling_cosmos3_omni.py).
// This creator therefore builds a `Qwen3VL` from the same understanding config.
//
// Loading is NOT wired here. A loader must run `SharpAICosmos3OmniSanitizer.sanitize`, then
// `qwen3VLAttentionNames`, then `Qwen3VL.sanitize`. The sanitizer rejects `*_moe_gen` keys, so a
// checkpoint that still carries the generation experts (nvidia/Cosmos3-Nano) does not load until
// the drop-or-reject policy is decided.
// Wiring into VLMTypeRegistry is left to the overlay consumer; VLMModelFactory is not edited.
//
// Known gaps (not implemented): generation experts (MoT, diffusion, action, sound), the processor
// for image and video inputs, and the subfolder weight layout (see SharpAICosmos3OmniWeightFiles).

import Foundation
import MLXLMCommon

public enum SharpAICosmos3OmniOverlay {

    /// Creators keyed by `model_type`, same closure shape as `VLMTypeRegistry`.
    nonisolated(unsafe) public static let creators: [String: (Data) throws -> any LanguageModel] = [
        "cosmos3_omni": { data in
            let configuration = try JSONDecoder.json5().decode(
                SharpAICosmos3OmniConfiguration.self, from: data)
            try configuration.validateModelConfiguration()
            let understanding = try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(from: data)
            return Qwen3VL(understanding)
        }
    ]
}
