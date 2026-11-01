// SharpAI fork overlay: step5_vl checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys keep their Hugging Face names. Only the precomputed rotary table is dropped.
// UNVERIFIED: no step5_vl checkpoint index is on disk. The fixture lists the PE tower as
// vision_model.transformer.resblocks.N.* and the projector as vit_large_projector.*; both pass
// through unchanged. MTP keys (model.layers.N.mtp_block.*, shared_head.*) also pass through,
// so the model body must skip them.

import Foundation
import MLX

public enum SharpAIStep5VlSanitize {

    /// Drops precomputed rotary tables. Other keys pass through unchanged.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
