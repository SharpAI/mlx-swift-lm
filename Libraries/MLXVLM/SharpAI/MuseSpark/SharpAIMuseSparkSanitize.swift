// SharpAI fork overlay: muse_spark checkpoint key sanitization. Not part of upstream.
//
// UNVERIFIED: no checkpoint index or weight key layout is available (the config is gated),
// so no keys are renamed or dropped. Replace this pass-through once the layout is known.

import Foundation
import MLX

public enum SharpAIMuseSparkSanitizer {

    /// Returns the checkpoint weights unchanged.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
