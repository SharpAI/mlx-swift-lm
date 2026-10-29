// Pure reference implementations for the #254 gap ops, built ONLY from existing MLX ops.
//
// Purpose: pin the spec (shapes, semantics, numerics) of ops the survey / kernel audit lists as
// "unknown" or "composable", so the later model implementation has an oracle to be checked
// against. Nothing here is used by shipping code; it is test support only.
//
// upstream-watch: this namespace is SharpAI-only and cannot collide with upstream names. If
// ml-explore/mlx-swift-lm lands DeepSeek-V4 compressor / indexer, GLM5Next kpool, MXFP4 checkpoint
// dequant or Qwen4Exp PLE, compare their output with these functions before retiring them.
//
// Provenance of each spec (vLLM snapshot dc45a5f; source not in repo, paths are inside that vLLM tree):
//   softmaxPoolCompress -> vllm/models/deepseek_v4/common/ops/{save_partial_states,fused_compress_quant_cache}.py
//   kpoolSoftmaxHadamard -> vllm/models/glm5next/nvidia/ops/kpool_compress.py
//   mxfp4Dequant        -> vllm/models/deepseek_v4/quant_config.py (expert_dtype "fp4" = MXFP4)
//   pleGate, pleDilatedShortConv -> vllm/models/qwen4_exp/common/ops/ple.py

import Foundation
import MLX

public enum SharpAIOverlayGapReference {

    // MARK: Rnj1 / gemma3_text final_logit_softcapping

    /// `cap * tanh(x / cap)` (Gemma2/3 final logit soft-capping).
    public static func tanhSoftcap(_ x: MLXArray, cap: Float) -> MLXArray {
        tanh(x / cap) * cap
    }

    // MARK: DeepSeek-V4 KV compressor (compress_ratio 4 with overlap, 128 without)

    /// For every COMPLETE block of `ratio` tokens (block b ends at position p = (b+1)*ratio - 1):
    ///   window = positions p - (1+overlap)*ratio + 1 ... p (positions < 0 are masked out)
    ///   score' = score + ape[pos % ratio]                      (ape added when the state is saved)
    ///   w      = softmax over the window, per channel
    ///   out    = sum(kv * w) over the window
    /// With `overlap` (ratio == 4) the state is `coff = 2` heads wide: the first `ratio` window
    /// tokens read channels [0, headDim), the last `ratio` read [headDim, 2*headDim).
    /// RMSNorm and RoPE are applied afterwards (see `rmsNorm`); both are omitted here.
    /// kv, score: [T, coff*headDim]; ape: [ratio, coff*headDim]; returns [T / ratio, headDim].
    public static func softmaxPoolCompress(
        kv: MLXArray, score: MLXArray, ape: MLXArray, ratio: Int, overlap: Bool, headDim: Int
    ) -> MLXArray {
        let blocks = kv.dim(0) / ratio
        precondition(blocks >= 1, "need at least one complete block")
        let window = (overlap ? 2 : 1) * ratio
        var rows = [MLXArray]()
        for b in 0 ..< blocks {
            let p = (b + 1) * ratio - 1
            let start = p - window + 1
            var ks = [MLXArray](), ss = [MLXArray]()
            for j in 0 ..< window {
                let pos = start + j
                if pos < 0 { continue }  // masked: -inf score and zero kv contribute nothing
                let off = (overlap && j >= ratio) ? headDim : 0
                ks.append(kv[pos, off ..< (off + headDim)])
                ss.append(score[pos, off ..< (off + headDim)] + ape[pos % ratio, off ..< (off + headDim)])
            }
            let w = softmax(stacked(ss), axis: 0)
            rows.append((stacked(ks) * w).sum(axis: 0))
        }
        return stacked(rows)
    }

    /// `x * rsqrt(mean(x^2) + eps) * weight` over the last axis.
    public static func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float) -> MLXArray {
        x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps) * weight
    }

    // MARK: GLM5Next kpool indexer

    /// Natural-order (Sylvester) Hadamard matrix of size n (power of two), +-1 entries.
    public static func sylvesterHadamard(_ n: Int) -> MLXArray {
        precondition(n > 0 && (n & (n - 1)) == 0)
        var rows: [[Float]] = [[1]]
        while rows.count < n {
            rows = rows.map { $0 + $0 } + rows.map { $0 + $0.map { -$0 } }
        }
        return MLXArray(rows.flatMap { $0 }, [n, n])
    }

    /// Orthonormal Hadamard rotation over the last axis (matches `_hadamard128`, scale 1/sqrt(n)).
    public static func hadamardRotate(_ x: MLXArray) -> MLXArray {
        let n = x.dim(-1)
        return matmul(x, sylvesterHadamard(n)) * (1 / Float(n).squareRoot())
    }

    /// kpool: one pool of `pool` consecutive tokens -> softmax(score + ape)-weighted sum of K
    /// (per channel, over the pool) -> Hadamard rotation. The fp8 absmax quantisation that the CUDA
    /// kernel fuses afterwards is NOT part of this reference.
    /// k, score: [pool, dim]; ape: [pool, dim]; returns [dim].
    public static func kpoolSoftmaxHadamard(k: MLXArray, score: MLXArray, ape: MLXArray) -> MLXArray {
        let w = softmax(score + ape, axis: 0)
        return hadamardRotate((k * w).sum(axis: 0))
    }

    // MARK: MXFP4 (E2M1 values, E8M0 group scales) checkpoint dequant

    public static let e2m1Magnitudes: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    /// packed: uint8 [rows, cols/2], element 2i in the LOW nibble, 2i+1 in the HIGH nibble;
    /// nibble = sign(bit 3) | magnitude index (bits 0-2). scales: uint8 [rows, cols/32] holding the
    /// E8M0 exponent, value = 2^(byte - 127). Returns float32 [rows, cols].
    public static func mxfp4Dequant(packed: MLXArray, scales: MLXArray, groupSize: Int = 32) -> MLXArray {
        let rows = packed.dim(0), cols = packed.dim(1) * 2
        let lo = packed.asType(.int32) & 15
        let hi = packed.asType(.int32) >> 4
        let codes = stacked([lo, hi], axis: -1).reshaped(rows, cols)
        let magnitude = take(MLXArray(e2m1Magnitudes), codes & 7)
        let value = which((codes & 8) .== 8, -magnitude, magnitude)
        let scale = pow(MLXArray(Float(2)), scales.asType(.float32) - 127)
        return (value.reshaped(rows, cols / groupSize, groupSize) * scale.expandedDimensions(axis: -1))
            .reshaped(rows, cols)
    }

    // MARK: Qwen4Exp PLE

    /// Gate half of the PLE layer. Per slot s of `hc` slots (each `h` wide):
    ///   k_n = rms(key_s) * (1 + normKey_s);  q_n = rms(hidden_s) * (1 + normQuery_s)
    ///   d = sum(k_n * q_n) / sqrt(h);  g = sigmoid(sign(d) * sqrt(max(|d|, 1e-6)))
    ///   gated_s = g * value;  normed_s = rms(gated_s) * (1 + normConv_s)
    /// key, hidden: [T, hc*h]; value: [T, h]; norm weights: [hc*h]. Returns (gated, normed) [T, hc*h].
    public static func pleGate(
        key: MLXArray, value: MLXArray, hidden: MLXArray,
        normKey: MLXArray, normQuery: MLXArray, normConv: MLXArray, eps: Float
    ) -> (gated: MLXArray, normed: MLXArray) {
        let t = hidden.dim(0), h = value.dim(-1), hc = hidden.dim(-1) / h
        func rms(_ x: MLXArray) -> MLXArray { x * rsqrt(x.square().mean(axis: -1, keepDims: true) + eps) }
        let k = key.reshaped(t, hc, h), q = hidden.reshaped(t, hc, h)
        let kn = rms(k) * (1 + normKey.reshaped(1, hc, h))
        let qn = rms(q) * (1 + normQuery.reshaped(1, hc, h))
        let d = (kn * qn).sum(axis: -1, keepDims: true) / Float(h).squareRoot()
        let sgn = which(d .< 0, MLXArray(Float(-1)), which(d .> 0, MLXArray(Float(1)), MLXArray(Float(0))))
        let g = sigmoid(sgn * sqrt(maximum(abs(d), MLXArray(Float(1e-6)))))
        let gated = g * value.reshaped(t, 1, h)
        let normed = rms(gated) * (1 + normConv.reshaped(1, hc, h))
        return (gated.reshaped(t, hc * h), normed.reshaped(t, hc * h))
    }

    /// Dilated causal depthwise short convolution followed by SiLU, zero history:
    ///   y[t, c] = silu( sum_k w[c, k] * x[t - (K-1-k)*dilation, c] )
    /// (decode state length = (K-1)*dilation). x: [T, C]; w: [C, K]. Returns [T, C].
    public static func pleDilatedShortConv(x: MLXArray, w: MLXArray, dilation: Int) -> MLXArray {
        let t = x.dim(0), kernel = w.dim(1)
        var acc = MLXArray.zeros(like: x)
        for k in 0 ..< kernel {
            let shift = (kernel - 1 - k) * dilation
            let tap: MLXArray
            if shift == 0 {
                tap = x
            } else if shift >= t {
                tap = MLXArray.zeros(like: x)
            } else {
                tap = concatenated([MLXArray.zeros([shift, x.dim(1)], dtype: x.dtype), x[0 ..< (t - shift)]], axis: 0)
            }
            acc = acc + tap * w[0..., k].reshaped(1, -1)
        }
        return acc * sigmoid(acc)
    }
}
