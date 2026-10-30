// SharpAI fork overlay: checkpoint-side mapping for DeepSeek-V4 `expert_dtype: "fp4"` (MXFP4) experts, and the
// `quantization_config` (scale_fmt ue8m0, weight_block_size) used by the FP8 block dequant. New file; no upstream
// counterpart (SL-35 step 5).
//
// Format (spec: vLLM deepseek_v4/quant_config.py, snapshot dc45a5f; not in this repo):
//   expert weight : 8-bit storage [rows, cols/2], element 2i in the LOW nibble, 2i+1 in the HIGH nibble, each nibble
//                   E2M1 (sign bit 3, magnitude bits 0-2)
//   expert scale  : 8-bit E8M0 exponent [rows, cols/32], value 2^(byte-127)
// MLX's mxfp4 layer stores exactly this: weight as uint32 words [rows, cols/8] (little-endian byte order = the same
// bytes), scales as uint8 [rows, cols/32], group size 32, 4 bits. So the mapping is a bit-preserving view plus the
// rename `.scale` -> `.scales`; no value is recomputed. `dequantized(mode: .mxfp4)` on the mapped tensors equals
// SharpAIOverlayGapReference.mxfp4Dequant exactly (SharpAIOverlayMXFP4Tests).
//
// Inferred, not verified against a real checkpoint: that the checkpoint tensor names are `<expert>.<proj>.weight` and
// `<expert>.<proj>.scale`, and that the tensors reach us as 8-bit integer arrays. Other dtypes are left untouched.
// upstream-watch: DeepseekV4.sanitize in mlx-swift-lm.

import Foundation
import MLX

extension SharpAIOverlay {

    /// Quantization parameters of the mapped expert layers: group 32, 4 bit, mode mxfp4.
    public static let mxfp4Quantization: BaseConfiguration.Quantization = {
        let json = Data(#"{"group_size":32,"bits":4,"mode":"mxfp4"}"#.utf8)
        return try! JSONDecoder().decode(BaseConfiguration.Quantization.self, from: json)
    }()

    /// Maps `...ffn.experts.<n>.<proj>.weight` / `.scale` pairs of an `expert_dtype == "fp4"` checkpoint onto the
    /// tensors MLX's mxfp4 quantized layers expect: the weight becomes uint32 [rows, cols/8], `.scale` is renamed
    /// `.scales` (uint8). Returns `weights` unchanged for any other `expertDtype`; a pair whose tensors are not
    /// 8-bit integers, or whose last dimension is not a multiple of 4 bytes, is left as it is.
    public static func mapMXFP4Experts(_ weights: [String: MLXArray], expertDtype: String?) -> [String: MLXArray] {
        guard expertDtype == "fp4" else { return weights }
        var out = weights
        for (key, scale) in weights where key.hasSuffix(".scale") && key.contains(".ffn.experts.") {
            let base = String(key.dropLast("scale".count))
            guard let weight = weights[base + "weight"], isByte(weight), isByte(scale),
                weight.ndim >= 1, weight.dim(-1) % 4 == 0
            else { continue }
            out[base + "weight"] = weight.asType(.uint8).view(dtype: .uint32)
            out[base + "scales"] = scale.asType(.uint8)
            out.removeValue(forKey: key)
        }
        return out
    }

    private static func isByte(_ a: MLXArray) -> Bool { a.dtype == .uint8 || a.dtype == .int8 }

    /// What the FP8 block dequant needs from a checkpoint's `quantization_config`.
    public struct FP8BlockConfig: Sendable, Equatable {
        /// Square block edge from `weight_block_size` ([128, 128] when absent).
        public var blockSize: Int = 128
        /// `scale_fmt == "ue8m0"`: integer-typed block scales are E8M0 exponent bytes.
        public var ue8m0: Bool = false

        public init(quantizationConfig: [String: JSONValue]?) {
            if case .array(let sizes)? = quantizationConfig?["weight_block_size"],
                case .int(let n)? = sizes.first, n > 0
            {
                blockSize = n
            }
            if case .string("ue8m0")? = quantizationConfig?["scale_fmt"] { ue8m0 = true }
        }
    }

    /// `weight * scaleInv` per `blockSize` x `blockSize` block (zero padded to whole blocks, cropped back).
    /// With `ue8m0`, a uint8/int8 `scaleInv` is decoded as 2^(byte-127) first; float scales are used as they are.
    /// Byte 0 decodes to 0 and byte 255 to infinity here (E8M0 reserves 255 for NaN); real scales avoid both.
    public static func dequantFP8Blocks(
        weight: MLXArray, scaleInv: MLXArray, blockSize bs: Int, ue8m0: Bool
    ) -> MLXArray {
        var scale = scaleInv
        if ue8m0, isByte(scaleInv) {
            scale = (scaleInv.asType(.uint8).asType(.uint32) << 23).view(dtype: .float32)
        }
        let (m, n) = (weight.dim(0), weight.dim(1))
        let padBottom = (bs - m % bs) % bs
        let padSide = (bs - n % bs) % bs
        var padded = MLX.padded(weight, widths: [.init((0, padBottom)), .init((0, padSide))])
        padded = padded.reshaped([(m + padBottom) / bs, bs, (n + padSide) / bs, bs])
        let scaled = padded * scale[0..., .newAxis, 0..., .newAxis]
        return scaled.reshaped([m + padBottom, n + padSide])[0 ..< m, 0 ..< n]
    }
}
