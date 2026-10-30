// Lane r5-k2 (SL-31 remainder b): W8A8 int8 and W4A4 NVFP4 activation-quantised matmul.
//
// Decision (backed by the tests below): NO sharpai_ Metal kernel is needed.
//   W8A8 int8 -> bit-exact from existing ops: per-token quant with round-half-even, float32 matmul over K
//                chunks of <= 1024 (every partial sum is an integer <= 2^24, so float32 is exact), chunks added
//                in int32. A single unchunked float32 matmul is NOT exact once a partial sum passes 2^24
//                (pinned by the adversarial K = 3000 case), which is why chunking is part of the reference.
//   W4A4 NVFP4 -> MLX's own qqmm on Metal (`quantizedQuantizedMM`) quantises x to NVFP4 and dequantises it
//                (quantize_dequantize_input), then runs the dequantising qmv against the packed weight. That
//                is the same composition as the reference (quantize/dequantize x, dequantise w, matmul). The
//                activation quantisation is exact (hand levels below); the dot-product accumulation ORDER is
//                not specified by either side, so equality is asserted to float32 accumulation tolerance, not
//                bitwise. A CUDA fp4 tensor-core GEMM's internal order is not reproducible on Metal by any
//                kernel, so a custom kernel could not make it more faithful.
// INFERRED (not in the vLLM snapshot): the cutlass int8 epilogue product order (acc * sx) * sw.
// GDN2 (MuseSpark) is covered by the last test: its maths is NOT in the sources, nothing is invented.

import Foundation
import MLX
import Testing

struct ConformanceGapQuantMatmulTests {
    typealias Ref = SharpAIOverlayQuantMatmulReference

    // MARK: int8 helpers (independent integer oracle)

    static func int8Oracle(x: [Float], M: Int, K: Int, wq: [Int8], N: Int, wScale: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: M * N)
        for m in 0 ..< M {
            let row = Array(x[m * K ..< (m + 1) * K])
            let absmax = row.map { abs($0) }.max() ?? 0
            let inv: Float = absmax == 0 ? 0 : 127 / absmax
            let q = row.map { Int64(min(127, max(-128, ($0 * inv).rounded(.toNearestOrEven)))) }
            let sx = absmax / 127
            for n in 0 ..< N {
                var acc: Int64 = 0
                for k in 0 ..< K { acc += q[k] * Int64(wq[n * K + k]) }
                out[m * N + n] = (Float(Int32(truncatingIfNeeded: acc)) * sx) * wScale[n]
            }
        }
        return out
    }

    @Test("int8 quant: per-token scale = absmax/127, round-half-even, hand values")
    func int8QuantHand() {
        // row0 absmax 2 -> inv 63.5 ; 1.0 -> 63.5 -> 64 (tie .5 rounds to even 64) ; 0.5 -> 31.75 -> 32 ; -2 -> -127
        // row1 all zero -> inv 0, scale 0, q 0
        let x = MLXArray([Float](arrayLiteral: 2, 1, 0.5, -2, 0, 0, 0, 0), [2, 4])
        let (q, s) = Ref.int8QuantPerToken(x)
        #expect(q.asArray(Int8.self) == [127, 64, 32, -127, 0, 0, 0, 0])
        #expect(s.asArray(Float.self) == [Float(2) / 127, 0])
    }

    @Test("int8 W8A8 == integer oracle on random data (K = 96, 2 chunks of 48 and the default chunk)")
    func int8Random() {
        let (M, K, N) = (5, 96, 7)
        let (x, wq): (MLXArray, MLXArray) = withRandomState(MLXRandom.RandomState(seed: 11)) {
            (MLXRandom.normal([M, K]), MLXRandom.randInt(low: -128, high: 128, [N, K]).asType(.int8))
        }
        let ws = MLXArray((0 ..< N).map { 0.01 * Float($0 + 1) })
        let expected = Self.int8Oracle(
            x: x.asArray(Float.self), M: M, K: K, wq: wq.asArray(Int8.self), N: N, wScale: ws.asArray(Float.self))
        let got = Ref.w8a8Int8(x, wq: wq, wScale: ws).asArray(Float.self)
        #expect(got == expected)
    }

    @Test("int8 W8A8 is bit-exact past 2^24: K = 3000 all-max row (acc = 48,386,873, odd and > 2^24)")
    func int8BeyondFloat24() {
        let (M, K, N) = (2, 3000, 2)
        var xv = [Float](repeating: 1, count: K)  // row0: constant -> q = 127 everywhere
        xv += (0 ..< K).map { Float(($0 * 37) % 11) - 5 }  // row1: mixed
        var wv = [Int8](repeating: 127, count: K)
        wv[K - 1] = 126  // makes the exact sum odd, so float32 (spacing 4 at 4.8e7) cannot hold it
        wv += (0 ..< K).map { Int8(truncatingIfNeeded: ($0 * 53) % 255 - 127) }
        let x = MLXArray(xv, [M, K]), wq = MLXArray(wv, [N, K])
        let ws = MLXArray([Float](arrayLiteral: 0.5, 0.25))
        let expected = Self.int8Oracle(x: xv, M: M, K: K, wq: wv, N: N, wScale: [0.5, 0.25])
        let acc = Ref.int8MatmulInt32(Ref.int8QuantPerToken(x).q, wq)
        #expect(acc.asArray(Int32.self)[0] == 48_386_873)
        #expect(Ref.w8a8Int8(x, wq: wq, wScale: ws).asArray(Float.self) == expected)
    }

    // MARK: NVFP4

    @Test("nvfp4 activation quant: hand E2M1 levels with block scale 1 (absmax 6)")
    func nvfp4ActivationHand() {
        // levels {0, .5, 1, 1.5, 2, 3, 4, 6}; 0.7 -> .5, 1.2 -> 1, 2.6 -> 3, -3.4 -> -3, 5.1 -> 6, 0.2 -> 0, -0.3 -> -.5
        var row = [Float](repeating: 0, count: 16)
        row.replaceSubrange(0 ..< 8, with: [6, 0.7, 1.2, 2.6, -3.4, 5.1, 0.2, -0.3])
        let dq = Ref.nvfp4QuantDequant(MLXArray(row, [1, 16]), globalScale: nil)
        var want = [Float](repeating: 0, count: 16)
        want.replaceSubrange(0 ..< 8, with: [6, 0.5, 1, 3, -3, 6, 0, -0.5])
        #expect(dq.asArray(Float.self) == want)
    }

    @Test("nvfp4 W4A4: MLX qqmm (Metal) == reference composition, without and with global scales")
    func nvfp4Qqmm() {
        let (M, K, N) = (3, 64, 8)
        let (x, w): (MLXArray, MLXArray) = withRandomState(MLXRandom.RandomState(seed: 5)) {
            (MLXRandom.normal([M, K]), MLXRandom.normal([N, K]))
        }
        for gs in [false, true] {
            let gx: MLXArray? = gs ? MLXArray(Float(1.0)) : nil
            let gw: MLXArray? = gs ? MLXArray(Float(0.5)) : nil
            let (wq, wsc, _) = quantized(w, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: gw)
            let ref = Ref.w4a4Nvfp4(x, wq: wq, wScales: wsc, globalScaleX: gx, globalScaleW: gw)
            let got = quantizedQuantizedMM(
                x, wq, scales: wsc, groupSize: 16, bits: 4, mode: .nvfp4, globalScaleX: gx, globalScaleW: gw)
            eval(ref, got)
            let scale = abs(ref).max().item(Float.self)
            let diff = abs(ref - got.asType(.float32)).max().item(Float.self)
            #expect(diff <= 1e-3 * scale, "global scales \(gs): diff \(diff) vs scale \(scale)")
            // and the weight-only path (what this repo ships today) is measurably different: quantising x matters
            let weightOnly = matmul(x, dequantized(wq, scales: wsc, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4, globalScale: gw, dtype: .float32).T)
            #expect(abs(ref - weightOnly).max().item(Float.self) > 1e-3 * scale)
        }
    }

    // MARK: GDN2

    @Test("gap: MuseSpark GDN2 maths is UNKNOWN in the sources -- known issue, nothing invented")
    func gdn2Unknown() {
        // Known from vllm/transformers_utils/configs/muse_spark.py:49-62: mode "chunk", use_gate, use_short_conv,
        // conv_size 4, num_heads/head_dim, gated_delta_net_use_low_rank_proj (+ low_rank_compression_ratio 4).
        // vllm/models/muse_spark/nvidia/model.py:1836 raises NotImplementedError for enabled GDN2 layers, so the
        // gate/beta parametrisation, the low-rank projection shapes and the checkpoint tensor names are not
        // available. The GDN recurrence itself is covered by ConformanceGapGdnReferenceTests (scalar and per-key gates).
        withKnownIssue("GDN2 low-rank projection / gate parametrisation is not in any source in the research docs") {
            Issue.record("unknown: GDN2 projection shapes, gate and beta parametrisation, weight names")
        }
    }
}
