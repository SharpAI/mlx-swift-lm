// SharpAI fork overlay: molmo2 checkpoint key sanitization. Not part of upstream mlx-swift-lm.
//
// Keys are kept as written in the checkpoint. No prefix remap is applied.
// UNVERIFIED: the vision-side prefixes below are not confirmed against a real molmo2 checkpoint
// index. Check them before relying on this sanitizer.

import Foundation
import MLX

public enum SharpAIMolmo2Sanitizer {

    static let droppedPrefixes = [
        "model.vision_backbone.",
        "vision_backbone.",
        "model.image_pooling_2d.",
        "image_pooling_2d.",
        "model.image_projector.",
        "image_projector.",
    ]

    /// Drops rotary buffers and vision-side weights. Language weights pass through unchanged.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            if droppedPrefixes.contains(where: { key.hasPrefix($0) }) {
                continue
            }
            sanitized[key] = value
        }

        return sanitized
    }
}
