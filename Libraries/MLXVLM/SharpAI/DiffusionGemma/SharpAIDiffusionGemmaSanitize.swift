// SharpAI fork overlay: diffusion_gemma checkpoint key sanitization. Not part of upstream.
//
// Checkpoint key names are verified against the weight index of google/diffusiongemma-26B-A4B-it
// (1047 keys; prefixes model.decoder.*, model.encoder.language_model.*, model.encoder.vision_tower.*,
// model.encoder.embed_vision.*). Verified against: https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/model.safetensors.index.json
//
// The index stores MoE experts as stacked tensors (`experts.gate_up_proj`, `experts.down_proj`,
// 3D, 128 experts per layer), so no per-expert split is needed. The index also has
// `model.encoder.language_model.layers.N.layer_scalar` only, with no encoder projection weights.
// The Swift module tree for these keys is not written yet, so this pass keeps every key unchanged.

import Foundation
import MLX

public enum SharpAIDiffusionGemmaSanitizer {

    /// Returns the weights with keys unchanged. Identity mapping until the Swift module tree exists;
    /// the key names it must accept are verified against the weight index (see the file header).
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        return weights
    }
}
