// SharpAI fork overlay: hunyuan_vl checkpoint key sanitization. Not part of upstream.
//
// Keeps HF key names, so module property names must match the checkpoint. This pass only drops
// precomputed rotary buffers.
// UNVERIFIED: the rotary buffer suffix is assumed. Confirm it against a real hunyuan_vl index.

import Foundation
import MLX

public enum SharpAIHunyuanVlSanitizer {

    /// Drops precomputed rotary keys. Other keys are unchanged.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.hasSuffix("rotary_emb.inv_freq") }
    }
}
