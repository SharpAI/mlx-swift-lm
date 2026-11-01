// SharpAI fork overlay: cohere_compass checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: the checkpoint key layout is not known. This only drops rotary inv_freq buffers,
// a convention shared by other models here. No vision tower key remap exists yet. The vision
// tower differs from Qwen3VL and the diff is not verified. Confirm against a real
// checkpoint index before adding remaps.

import Foundation
import MLX

public enum SharpAICohereCompassSanitizer {

    /// Removes non-weight buffers. All other keys keep their HF names.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
