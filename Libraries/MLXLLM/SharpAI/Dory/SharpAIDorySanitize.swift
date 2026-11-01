// SharpAI fork overlay: Dory checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// UNVERIFIED: no Dory checkpoint index or reference loader is in the repo, so no key is renamed
// or dropped. Keys keep their HF names until a real checkpoint confirms the mapping.

import Foundation
import MLX

public enum SharpAIDorySanitizer {

    /// Returns the weights unchanged. See the UNVERIFIED note above.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
