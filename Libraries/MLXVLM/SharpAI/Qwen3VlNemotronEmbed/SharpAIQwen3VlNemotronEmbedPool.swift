import Foundation
import MLX

/// Pools `[batch, seq, dim]` hidden states to `[batch, dim]` under a `[batch, seq]` mask.
/// `colbert` does not reduce over tokens. It returns `[batch, seq, dim]` with padded rows zeroed,
/// matching the reference `_extract_embeddings`. `embed` in the model does not call this helper.
public enum SharpAIQwen3VlNemotronEmbedPool {

    public static func pool(
        _ hidden: MLXArray, mask: MLXArray, mode: SharpAIQwen3VlNemotronEmbedPooling
    ) -> MLXArray {
        switch mode {
        case .colbert:
            return hidden * mask.asType(hidden.dtype).expandedDimensions(axis: -1)
        case .last:
            return lastToken(hidden, mask: mask)
        case .mean:
            return mean(hidden, mask: mask)
        }
    }

    /// The hidden state at the last position where `mask` is 1. A zero row when all of it is padding.
    static func lastToken(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let length = hidden.dim(1)
        let positions = MLXArray(0 ..< length).asType(.int32)[.newAxis, 0...]
        let lastPlusOne = (mask.asType(.int32) * (positions + MLXArray(Int32(1)))).max(axis: -1)
        let target = (lastPlusOne - MLXArray(Int32(1))).expandedDimensions(axis: -1)
        let selected = (positions .== target).asType(hidden.dtype).expandedDimensions(axis: -1)
        return (hidden * selected).sum(axis: 1)
    }

    /// Mean over real tokens only.
    static func mean(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        let weights = mask.asType(hidden.dtype).expandedDimensions(axis: -1)
        let total = (hidden * weights).sum(axis: 1)
        let count = maximum(weights.sum(axis: 1), MLXArray(Float(1)).asType(hidden.dtype))
        return total / count
    }
}
