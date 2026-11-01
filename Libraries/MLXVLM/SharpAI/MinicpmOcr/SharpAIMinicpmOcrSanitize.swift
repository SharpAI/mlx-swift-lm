// SharpAI fork overlay: minicpm_ocr checkpoint key sanitization. Not part of upstream mlx-swift-lm.

import Foundation
import MLX

/// Checkpoint key cleanup for `minicpm_ocr`. Only the non-persistent rotary buffer rule lives
/// here. The prefix remapping and the patch-embedding layout change are applied in
/// `SharpAIMinicpmOcr.sanitize(weights:)`, which calls this first.
public enum SharpAIMinicpmOcrSanitize {

    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}
