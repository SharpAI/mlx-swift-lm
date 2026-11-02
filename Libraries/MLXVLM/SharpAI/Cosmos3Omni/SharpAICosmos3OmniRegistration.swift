// SharpAI fork overlay: cosmos3_omni creator. Not part of upstream mlx-swift-lm.
//
// The understanding path is Qwen3-VL. In the transformers reference, `Cosmos3OmniModel` builds
// `visual` from `vision_config` and `language_model` from `text_config`, both through AutoModel
// (Verified against: https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/cosmos3_omni/modeling_cosmos3_omni.py).
// This creator therefore builds a `Qwen3VL` from the same understanding config.
//
// The creator returns `SharpAICosmos3OmniModel`: the Qwen3VL as child `und`, the generation tower
// (`gen`, the second Mixture-of-Transformers stream) and the modality heads (`gen_heads`). Its
// `sanitize(weights:)` runs the Cosmos sanitizer, `qwen3VLAttentionNames` and `Qwen3VL.sanitize`, so the
// real checkpoint keys (understanding, generation tower and heads) all load with none dropped.
// The aggregate in SharpAIVLMModelOverlays wires it into VLMTypeRegistry.
//
// Known gaps (not implemented): sampler and guidance, VAE, sound tokenizer, the processor for
// image and video inputs (the Qwen3VLProcessor is used, UNVERIFIED against the checkpoint), and
// the subfolder weight layout loading (see SharpAICosmos3OmniWeightFiles).

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
            // The head sizes are decoded on their own, without the tower checks (the tiny reference
            // config has an mrope_section that does not sum to head_dim / 2, which only matters for
            // the tower). A malformed head field throws here instead of building default-sized heads.
            let heads = try SharpAICosmos3OmniHeadsConfiguration.decode(
                rootConfig: data, fallbackHiddenSize: understanding.textConfiguration.hiddenSize)
            return SharpAICosmos3OmniModel(understanding, heads: heads)
        }
    ]
}
