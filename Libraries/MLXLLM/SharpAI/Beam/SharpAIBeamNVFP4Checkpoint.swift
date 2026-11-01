import Foundation
import MLX

/// NVFP4 (W4A4) checkpoint loading for Beam. ModelOpt-style tensors are rewritten to MLX names.
///
/// UNVERIFIED: the checkpoint tensor layout is taken from vLLM `modelopt.py`. The byte order of
/// the packed codes, the dtype of `weight_scale` in the safetensors reader, and the meaning of
/// `global_scale` against ModelOpt `weight_scale_2` are not checked against a real Beam shard.
public enum SharpAIBeamNVFP4Checkpoint {
    public static let groupSize = 16
    public static let bits = 4
    public static let mode: QuantizationMode = .nvfp4

    /// Rewrites ModelOpt NVFP4 tensors into the names MLX's nvfp4 ops use.
    ///
    /// - `<p>.weight` (uint8, two codes per byte) is viewed as uint32 when a `<p>.weight_scale` exists.
    /// - `<p>.weight_scale` becomes `<p>.scales` (uint8 bit pattern).
    /// - `<p>.weight_scale_2` becomes `<p>.global_scale` (float32).
    /// - `<p>.input_scale` becomes `<p>.input_global_scale` (float32), the static activation scale.
    public static func normalize(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var output = [String: MLXArray]()
        output.reserveCapacity(weights.count)

        for (key, value) in weights {
            if let base = stripSuffix(key, ".weight_scale_2") {
                output["\(base).global_scale"] = value.asType(.float32)
            } else if let base = stripSuffix(key, ".weight_scale") {
                output["\(base).scales"] = value.dtype == .uint8 ? value : value.view(dtype: .uint8)
            } else if let base = stripSuffix(key, ".input_scale") {
                output["\(base).input_global_scale"] = value.asType(.float32)
            } else if let base = stripSuffix(key, ".weight"),
                weights["\(base).weight_scale"] != nil
            {
                guard value.dtype == .uint8 else {
                    throw SharpAIBeamError("NVFP4 weight \(key) must be uint8 packed codes.")
                }
                output[key] = value.view(dtype: .uint32)
            } else {
                output[key] = value
            }
        }
        return output
    }

    /// W4A4 linear: quantizes `x` to NVFP4 on the fly and multiplies by packed NVFP4 weights.
    ///
    /// `x` is `[M, K]`, the packed weight is `[N, K / 8]` uint32, and the result is `[M, N]`
    /// (`x @ w.T`, the same orientation as `Linear`).
    /// UNVERIFIED: this helper is not yet wired into the attention or MLP modules. It is the
    /// op the loaded tensors feed.
    public static func w4a4Linear(
        _ x: MLXArray,
        weight: MLXArray,
        scales: MLXArray,
        globalScale: MLXArray?,
        inputGlobalScale: MLXArray?
    ) -> MLXArray {
        quantizedQuantizedMM(
            x, weight, scales: scales,
            groupSize: groupSize, bits: bits, mode: mode,
            globalScaleX: inputGlobalScale, globalScaleW: globalScale)
    }

    private static func stripSuffix(_ key: String, _ suffix: String) -> String? {
        guard key.hasSuffix(suffix) else { return nil }
        return String(key.dropLast(suffix.count))
    }
}
