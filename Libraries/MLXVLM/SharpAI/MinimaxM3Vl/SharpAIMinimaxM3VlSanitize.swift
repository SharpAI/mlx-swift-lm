// SharpAI fork overlay: minimax_m3_vl checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Pass-through: keys keep their HF names.
// UNVERIFIED: no M3 checkpoint or source defines the key layout, so nothing is remapped or dropped.

import Foundation
import MLX

public enum SharpAIMinimaxM3VlSanitizer {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
