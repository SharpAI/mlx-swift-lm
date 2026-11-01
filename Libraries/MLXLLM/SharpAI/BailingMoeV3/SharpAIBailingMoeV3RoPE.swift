//
//  SharpAIBailingMoeV3RoPE.swift
//  MLXLLM
//
//  Bailing M-RoPE for the MLA rope slice (vLLM bailing_mrope.py). Pairs are
//  GPT-J style (even, odd). Frequency j < H+W reads the height position when
//  j is even and the width position when j is odd. Other frequencies read the
//  temporal position. For text, T = H = W, so this equals plain interleaved RoPE.
//

import Foundation
import MLX
import MLXLMCommon

struct SharpAIBailingMoeV3RoPE {
    let dims: Int
    let invFreq: MLXArray
    let temporalMask: MLXArray
    let heightMask: MLXArray
    let widthMask: MLXArray

    /// `mropeSection` is [T, H, W]. Pass nil for plain text RoPE.
    init(dims: Int, base: Float, mropeSection: [Int]?) {
        let half = dims / 2
        let spatial: Int
        if let section = mropeSection, section.count == 3 {
            spatial = section[1] + section[2]
        } else {
            spatial = 0
        }
        self.dims = dims
        self.invFreq = MLXArray(
            (0 ..< half).map { Float(pow(Double(base), -2.0 * Double($0) / Double(dims))) })
        self.temporalMask = MLXArray((0 ..< half).map { $0 >= spatial ? Float(1) : Float(0) })
        self.heightMask = MLXArray(
            (0 ..< half).map { $0 < spatial && $0 % 2 == 0 ? Float(1) : Float(0) })
        self.widthMask = MLXArray(
            (0 ..< half).map { $0 < spatial && $0 % 2 == 1 ? Float(1) : Float(0) })
    }

    /// Text path: the same positions for T, H and W. `x` is [B, H, L, dims].
    func callAsFunction(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        let length = x.dim(2)
        let steps = MLXArray((0 ..< length).map { Float($0) })
        let positions: MLXArray
        switch offset {
        case .none:
            positions = steps.reshaped(1, length)
        case .some(.scalar(let value)):
            positions = (steps + Float(value)).reshaped(1, length)
        case .some(.batch(let value)):
            positions = value.asType(.float32).reshaped(-1, 1) + steps.reshaped(1, length)
        }
        return apply(x, temporal: positions, height: positions, width: positions)
    }

    /// Each position array is [B or 1, L] float32.
    func apply(
        _ x: MLXArray, temporal: MLXArray, height: MLXArray, width: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)
        let heads = x.dim(1)
        let length = x.dim(2)
        let half = dims / 2

        // Masks are disjoint, so each frequency reads exactly one axis.
        let selected =
            temporal.expandedDimensions(axis: -1) * temporalMask
            + height.expandedDimensions(axis: -1) * heightMask
            + width.expandedDimensions(axis: -1) * widthMask
        let angles = selected * invFreq
        let cosines = MLX.cos(angles).asType(x.dtype).expandedDimensions(axis: 1)
        let sines = MLX.sin(angles).asType(x.dtype).expandedDimensions(axis: 1)

        let pairs = x.reshaped(batch, heads, length, half, 2)
        let parts = split(pairs, indices: [1], axis: -1)
        let even = MLX.squeezed(parts[0], axis: -1)
        let odd = MLX.squeezed(parts[1], axis: -1)

        let rotatedEven = even * cosines - odd * sines
        let rotatedOdd = odd * cosines + even * sines
        return MLX.stacked([rotatedEven, rotatedOdd], axis: -1)
            .reshaped(batch, heads, length, dims)
    }
}
