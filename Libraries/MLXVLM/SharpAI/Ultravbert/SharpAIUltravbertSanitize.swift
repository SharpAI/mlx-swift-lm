// SharpAI fork overlay: UltraVBERT weight sanitize. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

public enum SharpAIUltravbertSanitize {

    /// UNVERIFIED: no checkpoint key names are known yet, so no remapping is applied.
    /// Keys pass through unchanged until the encoder layout is confirmed.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
