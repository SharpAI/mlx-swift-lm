// SharpAI fork overlay: minicpmv4_7 standalone key helper. Not part of upstream mlx-swift-lm.
//
// The checkpoint key remap for the model lives in SharpAIMinicpmv47.sanitize(weights:) and
// SharpAIMinicpmv47.remapKey(_:), which are checked against the HF key layout. This helper is a
// passthrough and does not rename anything.

import Foundation
import MLX

public enum SharpAIMinicpmv47Sanitizer {

    /// Returns the weights with their checkpoint keys unchanged.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
