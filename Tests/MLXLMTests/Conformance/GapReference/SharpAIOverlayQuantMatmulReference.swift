// Pure reference implementations of activation-quantised matmul (W8A8 int8, W4A4 NVFP4), built ONLY from
// existing MLX ops. Lane r5-k2 (SL-31 remainder). Test support only; nothing here ships.
//
// upstream-watch: SharpAI-only namespace. If ml-explore/mlx-swift(-lm) lands an int8 W8A8 linear, compare
// against `w8a8Int8` before retiring this.
//
// Provenance (vLLM snapshot under scratchpad/survey/vllm_src, read-only):
//   int8 dynamic per-token quant -> csrc/quantization/w8a8/int8/scaled_quant.cu:143-171
//       scale_t = absmax / 127 ; inv = (absmax == 0) ? 0 : 127 / absmax ; q = cvt.rni.sat.s8(x * inv)
//       (round-to-nearest-EVEN, saturate to [-128, 127]). The Triton variant in
//       model_executor/layers/quantization/utils/int8_utils.py:52 floors absmax at 1e-10 and rounds
//       half-away; the CUDA kernel is taken as the reference, the Triton difference is only on exact .5 ties.
//   int8 GEMM: weights int8 [N,K] with a per-output-channel fp32 scale (compressed_tensors_w8a8_int8.py,
//       strategy CHANNEL), int32 accumulation, epilogue acc * scale_x * scale_w. INFERRED: the cutlass
//       epilogue is not in the snapshot; the product order (acc * sx) * sw is assumed.
//   NVFP4 W4A4: compressed_tensors_w4a4_nvfp4.py + kernels/linear/nvfp4/pytorch.py -- activations are
//       quantised on the fly (group 16, e4m3 block scale, global scale), out = fp4-gemm * alpha.
//       MLX's own `quantizedQuantizedMM` implements the same contract (see the tests).

import Foundation
import MLX

public enum SharpAIOverlayQuantMatmulReference {

    // MARK: W8A8 int8

    /// Dynamic symmetric per-token (per row) int8 quantisation. x: [M, K]. Returns q int8 [M, K], scale f32 [M, 1].
    public static func int8QuantPerToken(_ x: MLXArray) -> (q: MLXArray, scale: MLXArray) {
        let x32 = x.asType(.float32)
        let absmax = abs(x32).max(axis: -1, keepDims: true)
        let inv = which(absmax .== 0, MLXArray(Float(0)), 127 / maximum(absmax, 1e-30))
        // MLX `round` is half-to-even, matching cvt.rni.
        let q = clip(round(x32 * inv), min: -128, max: 127).asType(.int8)
        return (q, absmax / 127)
    }

    /// Exact int32 accumulation of `q [M,K] int8` times `w [N,K] int8` transposed, as int32 [M,N].
    /// A float32 matmul is exact for integer data while every partial sum is <= 2^24, i.e. K <= 1024
    /// (|q|,|w| <= 128 -> 128*128*1024 = 2^24), so K is processed in chunks of 1024 and the chunks are
    /// added in int32.
    public static func int8MatmulInt32(_ q: MLXArray, _ w: MLXArray, chunk: Int = 1024) -> MLXArray {
        let k = q.dim(-1)
        var acc: MLXArray? = nil
        for start in stride(from: 0, to: k, by: chunk) {
            let end = min(start + chunk, k)
            let part = matmul(
                q[0..., start ..< end].asType(.float32), w[0..., start ..< end].asType(.float32).T
            ).asType(.int32)
            acc = acc.map { $0 + part } ?? part
        }
        return acc!
    }

    /// Full W8A8 linear: quantise x per token, int32 GEMM against int8 weights, rescale.
    /// wScale: [N] or [N,1] fp32 (per output channel).
    public static func w8a8Int8(_ x: MLXArray, wq: MLXArray, wScale: MLXArray) -> MLXArray {
        let (q, sx) = int8QuantPerToken(x)
        let acc = int8MatmulInt32(q, wq)
        return (acc.asType(.float32) * sx) * wScale.reshaped(1, -1).asType(.float32)
    }

    // MARK: W4A4 NVFP4

    /// On-the-fly NVFP4 activation quantisation, returned dequantised in float32 (this is exactly what the
    /// fp4 tensor-core GEMM consumes, expressed in a wider type). x: [M, K], K % 16 == 0.
    public static func nvfp4QuantDequant(_ x: MLXArray, globalScale: MLXArray?) -> MLXArray {
        let (wq, s, _) = quantized(x, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: globalScale)
        return dequantized(
            wq, scales: s, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
            globalScale: globalScale, dtype: .float32)
    }

    /// W4A4: dequantised-activation times dequantised-weight, float32 accumulate.
    public static func w4a4Nvfp4(
        _ x: MLXArray, wq: MLXArray, wScales: MLXArray, globalScaleX: MLXArray?, globalScaleW: MLXArray?
    ) -> MLXArray {
        let xd = nvfp4QuantDequant(x, globalScale: globalScaleX)
        let wd = dequantized(
            wq, scales: wScales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4,
            globalScale: globalScaleW, dtype: .float32)
        return matmul(xd, wd.T)
    }
}
