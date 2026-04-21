import Foundation
import MLX
import MLXNN

/// Fixes an MLXNN Power primitive compilation crash on M1/M2/M3 GPUs.
/// The original `geluApproximate` uses `x ** 3`, which maps to the Power primitive.
/// Power lacks `output_shapes` under `compile(shapeless: true)` and returns 0 outputs on compilation,
/// causing a crash when `compileState.call([a])[0]` is executed.
/// This safe replacement uses `x * x * x` which maps to Multiply (supported natively).
///
/// Declared as `compile(shapeless: true)` so MLX caches and reuses the compiled Metal kernel
/// across all call sites instead of re-tracing the graph on every invocation.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray =
    compile(shapeless: true) { (x: MLXArray) -> MLXArray in
        let xFloat = x.asType(.float32)
        let half = MLXArray(0.5, dtype: .float32)
        let one = MLXArray(1.0, dtype: .float32)
        let c1 = MLXArray(Float(sqrt(2 / Float.pi)), dtype: .float32)
        let c2 = MLXArray(0.044715, dtype: .float32)
        let out = half * xFloat * (one + tanh(c1 * (xFloat + c2 * xFloat * xFloat * xFloat)))
        return out.asType(x.dtype)
    }

/// Module wrapper for `safeGeluApproximate` — drop-in for `GELU(approximation: .precise)`.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Fuses logit softcap operations (`tanh(x / cap) * cap`) into a single Metal dispatch.
private let _compiledSoftcapFn = MLX.compile(shapeless: true) { (args: [MLXArray]) -> [MLXArray] in
    [tanh(args[0] / args[1]) * args[1]]
}

public func compiledSoftcap(x: MLXArray, cap: MLXArray) -> MLXArray {
    return _compiledSoftcapFn([x, cap])[0]
}
