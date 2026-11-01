// SharpAI fork overlay: minimax_m3_dspark checkpoint key sanitize. Not part of upstream mlx-swift-lm.
//
// Pass-through. UNVERIFIED: no draft checkpoint or source defines the key layout, so nothing is
// renamed or dropped. DSpark drafters often share embed_tokens and lm_head with the target, but
// that is not confirmed for this model.

import Foundation
import MLX

public enum SharpAIMinimaxM3DsparkSanitizer {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
