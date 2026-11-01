// SharpAI fork overlay: pixel shuffle and MLP projector for the LlamaNemotronVL embedder. Not part of upstream mlx-swift-lm.

import Foundation
import MLX
import MLXNN

enum SharpAILlamaNemotronVlProjector {

    /// Module path `mlp1.*`. Checkpoint keys `mlp1.0`, `mlp1.1`, `mlp1.3` are renamed in sanitize.
    final class MLP1: Module {
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
            self.norm = LayerNorm(dimensions: inputDimensions, eps: 1e-5)
            self.fc1 = Linear(inputDimensions, hiddenDimensions, bias: true)
            self.fc2 = Linear(hiddenDimensions, outputDimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(norm(x))))
        }
    }

    /// Groups each `factor x factor` block of tokens into one token, as vLLM's `pixel_shuffle`.
    ///
    /// - Parameters:
    ///   - x: `[batch, width, height, channels]` (NHWC, square grid).
    ///   - factor: `k` with `downsample_ratio == 1 / k`.
    ///   - version: `"v1"` skips the second transpose. Any other value uses it.
    /// - Returns: `[batch, width / k, height / k, channels * k * k]`.
    static func pixelShuffle(_ x: MLXArray, factor k: Int, version: String) -> MLXArray {
        let (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var y = x.reshaped(n, w, h / k, c * k)
        y = y.transposed(0, 2, 1, 3)
        y = y.reshaped(n, h / k, w / k, c * k * k)
        if version != "v1" {
            y = y.transposed(0, 2, 1, 3)
        }
        return y
    }
}
