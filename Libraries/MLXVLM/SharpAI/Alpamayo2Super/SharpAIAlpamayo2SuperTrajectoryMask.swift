// SharpAI fork overlay: alpamayo2_super trajectory-token logit mask. Not part of upstream mlx-swift-lm.
//
// Verified against: https://github.com/NVlabs/alpamayo2/blob/main/src/alpamayo2_super/models/alpamayo2_super.py
// MaskDiscreteTrajectoryLogitsProcessor sets the trajectory-token span to -inf during CoT generation
// (sample_trajectories_from_data only). The training forward pass does not apply it.

import Foundation
import MLX

public enum SharpAIAlpamayo2SuperTrajectoryMask {

    /// Additive mask over the vocabulary: 0 for ordinary tokens, -inf for trajectory tokens.
    public static func additiveMask(
        vocabularySize: Int, trajectoryTokens: Range<Int>, dtype: DType = .float32
    ) throws -> MLXArray {
        guard vocabularySize > 0, !trajectoryTokens.isEmpty,
            trajectoryTokens.lowerBound >= 0, trajectoryTokens.upperBound <= vocabularySize
        else {
            throw SharpAIAlpamayo2SuperError.invalidTrajectoryRange(
                "\(trajectoryTokens) does not fit vocab size \(vocabularySize)")
        }
        var values = [Float](repeating: 0, count: vocabularySize)
        for id in trajectoryTokens {
            values[id] = -Float.infinity
        }
        return MLXArray(values, [vocabularySize]).asType(dtype)
    }

    /// Adds the mask to `logits` along the last (vocabulary) axis.
    public static func apply(_ logits: MLXArray, trajectoryTokens: Range<Int>) throws -> MLXArray {
        let mask = try additiveMask(
            vocabularySize: logits.dim(-1), trajectoryTokens: trajectoryTokens,
            dtype: logits.dtype)
        return logits + mask
    }
}
