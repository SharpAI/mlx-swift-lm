import Foundation
import MLX

/// Fixes an MLXNN Power primitive compilation crash on M1/M2/M3 GPUs.
/// The original `geluApproximate` uses `x ** 3`, which maps to the Power primitive.
/// Power lacks `output_shapes` under `compile(shapeless: true)` and returns 0 outputs on compilation,
/// causing a crash when `compileState.call([a])[0]` is executed.
/// This safe replacement uses `x * x * x` which maps to Multiply (supported natively).
public func safeGeluApproximate(_ x: MLXArray) -> MLXArray {
    let sqrt2OverPi = MLXArray(Float(sqrt(2.0 / .pi)), dtype: x.dtype)
    return 0.5 * x * (1.0 + tanh(sqrt2OverPi * (x + 0.044715 * (x * x * x))))
}

/// Fuses logit softcap operations (`tanh(x / cap) * cap`) into a single Metal dispatch.
private let _compiledSoftcapFn = MLX.compile(shapeless: true) { (args: [MLXArray]) -> [MLXArray] in
    [tanh(args[0] / args[1]) * args[1]]
}

public func compiledSoftcap(x: MLXArray, cap: MLXArray) -> MLXArray {
    return _compiledSoftcapFn([x, cap])[0]
}
