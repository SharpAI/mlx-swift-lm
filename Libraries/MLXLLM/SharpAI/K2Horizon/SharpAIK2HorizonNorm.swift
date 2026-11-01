// SharpAI fork overlay: grouped RMSNorm for k2_horizon. Not part of upstream mlx-swift-lm.
//
// The last axis is split into `groups` equal parts. Each part is RMS-normalized on its own,
// then scaled by one weight of the full width. With groups = 1 this is plain RMSNorm.
//
// UNVERIFIED: which norms use grouping and the group layout (contiguous split) are assumed.
// No reference implementation was available to check them.

import Foundation
import MLX
import MLXNN

public class SharpAIK2HorizonGroupedRMSNorm: Module, UnaryLayer {
    let weight: MLXArray
    let groups: Int
    let eps: Float

    public init(dimensions: Int, groups: Int, eps: Float) {
        precondition(groups > 0 && dimensions % groups == 0)
        self.weight = MLXArray.ones([dimensions])
        self.groups = groups
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let width = x.dim(-1)
        let leading = Array(x.shape.dropLast())
        let grouped = x.reshaped(leading + [groups, width / groups])
        let normalized = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
        return normalized.reshaped(x.shape) * weight
    }
}
