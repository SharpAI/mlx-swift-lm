// SharpAI fork overlay: ming_v3_asr checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: no Ming-3.0-asr checkpoint index is on disk, so no key remapping is known.
// Keys pass through unchanged. Replace this once the real key names are read.

import Foundation
import MLX

public enum SharpAIMingV3AsrSanitizer {

    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
