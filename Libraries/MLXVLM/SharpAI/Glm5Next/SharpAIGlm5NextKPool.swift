// SharpAI fork overlay: glm5_next kpool compression op. Not part of upstream mlx-swift-lm.
//
// Pure op only. Same math as SharpAIOverlayGapReference.kpoolSoftmaxHadamard in the test
// support (vLLM glm5next kpool_compress.py, snapshot dc45a5f; source not in repo).
// UNVERIFIED: how positions are grouped into pools, the tail selection (index_kpool_always_select_tail)
// and the fp8 quantization after the rotation are not implemented here.

import Foundation
import MLX

public enum SharpAIGlm5NextKPool {

    /// Natural-order (Sylvester) Hadamard matrix of size n (power of two), +-1 entries.
    public static func sylvesterHadamard(_ n: Int) -> MLXArray {
        precondition(n > 0 && (n & (n - 1)) == 0, "Hadamard size must be a power of two")
        var rows: [[Float]] = [[1]]
        while rows.count < n {
            let top = rows.map { row in row + row }
            let bottom = rows.map { row in row + row.map { -$0 } }
            rows = top + bottom
        }
        return MLXArray(rows.flatMap { $0 }, [n, n])
    }

    /// Orthonormal Hadamard rotation over the last axis.
    public static func hadamardRotate(_ x: MLXArray) -> MLXArray {
        let n = x.dim(-1)
        return matmul(x, sylvesterHadamard(n)) * (1 / Float(n).squareRoot())
    }

    /// One pool of tokens to one compressed key.
    /// k, score, ape: [pool, dim]. Returns [dim].
    public static func compress(k: MLXArray, score: MLXArray, ape: MLXArray) -> MLXArray {
        let weights = softmax(score + ape, axis: 0)
        return hadamardRotate((k * weights).sum(axis: 0))
    }
}
