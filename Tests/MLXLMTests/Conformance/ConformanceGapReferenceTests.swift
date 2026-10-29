// Conformance gap pins, part 3: reference implementations of #254 "composable / unknown" ops.
//
// These are GREEN tests of the reference code in GapReference/ against tiny cases whose answers
// were derived independently of the MLX code (by hand, or with python float64 where noted). They
// pin the spec so the later model implementation has an oracle; they do not test shipping code.
//
// Resolved status of the five ops the kernel audit (D1) left as "unknown" -- evidence is in the
// header of GapReference/SharpAIOverlayGapReference.swift and in scripts/conformance/README.md:
//   V4 KV compressor        -> composable (softmax-pool over a window; reshape/softmax/sum + RMSNorm + RoPE)
//   GLM5Next kpool indexer  -> composable (same softmax-pool, then a Hadamard-128 rotation; MLX has hadamardTransform)
//   DeepSeek-V4 FP4 experts -> composable (MXFP4: E2M1 + E8M0 group-32, MLX has the mxfp4 mode)
//   Qwen4Exp PLE            -> composable (gate = norms + dot + sigmoid; conv = dilated depthwise conv + silu)
//   MuseSpark GDN2          -> NOT RESOLVABLE from vLLM: its native stack raises NotImplementedError for it;
//                              the config is plain GatedDeltaNet (+ optional low-rank proj), so likely
//                              composable on the existing GatedDelta.swift, but there is no reference to pin.

import Foundation
import MLX
import Testing

struct ConformanceGapReferenceTests {
    typealias Ref = SharpAIOverlayGapReference

    static func values(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }

    static func expectClose(_ got: [Float], _ want: [Float], tol: Float = 1e-4, _ what: String = "") {
        #expect(got.count == want.count, "\(what): count \(got.count) vs \(want.count)")
        for (g, w) in zip(got, want) { #expect(abs(g - w) <= tol, "\(what): got \(g), want \(w)") }
    }

    @Test("tanhSoftcap: tanh(1)*cap at x == cap, odd, bounded by cap")
    func softcapReference() {
        let x = MLXArray([Float(0), 5, -5, 1000, -1000])
        // tanh(1) = 0.7615941559557649 (hand value); x = +-1000 saturates to +-cap.
        Self.expectClose(Self.values(Ref.tanhSoftcap(x, cap: 5)), [0, 3.8079708, -3.8079708, 5, -5])
    }

    // V4 compressor ------------------------------------------------------------------------

    @Test("V4 compressor, no overlap: ape-biased softmax pooling per complete block")
    func compressorNoOverlap() {
        // ratio 2, head 1. ape = [0, ln 3] -> in-block weights [1/4, 3/4] for every block.
        // block 0: .25*1 + .75*3 = 2.5 ; block 1: .25*5 + .75*7 = 6.5
        let kv = MLXArray([Float(1), 3, 5, 7], [4, 1])
        let score = MLXArray.zeros([4, 1])
        let ape = MLXArray([Float(0), Float(log(3.0))], [2, 1])
        let out = Ref.softmaxPoolCompress(kv: kv, score: score, ape: ape, ratio: 2, overlap: false, headDim: 1)
        #expect(out.shape == [2, 1])
        Self.expectClose(Self.values(out), [2.5, 6.5])
    }

    @Test("V4 compressor, overlap (ratio 4 style): first block masks negative positions, later blocks mix both channel halves")
    func compressorOverlap() {
        // ratio 2, head 1, coff 2 (state 2 wide). Uniform scores. ch0 = 10,20,30,40 ; ch1 = 1,2,3,4.
        // block 0 (p=1): window pos -2..1 -> only pos 0,1 valid, both in the 2nd half -> ch1: mean(1,2) = 1.5
        // block 1 (p=3): window pos 0..3 -> pos 0,1 from ch0 (10,20), pos 2,3 from ch1 (3,4) -> 37/4 = 9.25
        let kv = MLXArray([Float(10), 1, 20, 2, 30, 3, 40, 4], [4, 2])
        let out = Ref.softmaxPoolCompress(
            kv: kv, score: MLXArray.zeros([4, 2]), ape: MLXArray.zeros([2, 2]), ratio: 2, overlap: true, headDim: 1)
        Self.expectClose(Self.values(out), [1.5, 9.25])
    }

    @Test("rmsNorm hand case: [3,4] / sqrt(12.5)")
    func rmsNormReference() {
        let out = Ref.rmsNorm(MLXArray([Float(3), 4]), weight: MLXArray([Float(1), 1]), eps: 0)
        Self.expectClose(Self.values(out), [0.8485281, 1.1313708])  // python float64: 3/sqrt(12.5), 4/sqrt(12.5)
    }

    // GLM5Next kpool -------------------------------------------------------------------------

    @Test("kpool: pooling then Hadamard-128 of a unit vector is the (+-1)/sqrt(128) Hadamard column")
    func kpoolHandCase() {
        // pool 2, dim 128; uniform scores -> pooled = mean(k0, k1) = e0 (k0 = 2*e0, k1 = 0).
        // Hadamard column 0 is all ones -> every entry 1/sqrt(128) = 0.08838834764831845.
        var k0 = [Float](repeating: 0, count: 128); k0[0] = 2
        let k = stacked([MLXArray(k0), MLXArray.zeros([128])])
        let out = Ref.kpoolSoftmaxHadamard(k: k, score: MLXArray.zeros([2, 128]), ape: MLXArray.zeros([2, 128]))
        Self.expectClose(Self.values(out), [Float](repeating: 0.08838835, count: 128), tol: 1e-6)
        // e1 -> column 1 of the natural-order Hadamard matrix alternates +,-,+,-,...
        var k1 = [Float](repeating: 0, count: 128); k1[1] = 2
        let out1 = Ref.kpoolSoftmaxHadamard(
            k: stacked([MLXArray(k1), MLXArray.zeros([128])]),
            score: MLXArray.zeros([2, 128]), ape: MLXArray.zeros([2, 128]))
        let want = (0 ..< 128).map { Float($0 % 2 == 0 ? 0.08838835 : -0.08838835) }
        Self.expectClose(Self.values(out1), want, tol: 1e-6)
    }

    @Test("kpool Hadamard reference agrees with MLX hadamardTransform on random input")
    func hadamardMatchesMLX() {
        MLXRandom.seed(7)
        let x = MLXRandom.normal([3, 128])
        let a = Ref.hadamardRotate(x)
        let b = hadamardTransform(x)  // default scale 1/sqrt(n)
        let diff = abs(a - b).max().item(Float.self)
        #expect(diff < 1e-4, "max |ref - MLX| = \(diff)")
    }

    // MXFP4 --------------------------------------------------------------------------------

    @Test("MXFP4 dequant hand case: E2M1 nibbles (low first), E8M0 group scales")
    func mxfp4HandCase() {
        // 1 row x 64 cols = 2 groups of 32. Group 0 scale byte 128 -> 2^1 = 2. Group 1 byte 126 -> 0.5.
        // bytes 0x21 -> [0.5, 1.0] ; 0xF9 -> [-0.5 (9 = sign|1), -6 (F = sign|7)] ; 0x76 -> [4, 6]
        // group 0 first six: x2 -> [1, 2, -1, -12, 8, 12]. Group 1 byte 0x42 -> [1, 2] x0.5 -> [0.5, 1].
        var bytes = [UInt8](repeating: 0, count: 32)
        bytes[0] = 0x21; bytes[1] = 0xF9; bytes[2] = 0x76; bytes[16] = 0x42
        let out = Ref.mxfp4Dequant(packed: MLXArray(bytes, [1, 32]), scales: MLXArray([UInt8(128), 126], [1, 2]))
        #expect(out.shape == [1, 64])
        let v = Self.values(out)
        Self.expectClose(Array(v[0 ..< 6]), [1, 2, -1, -12, 8, 12], tol: 0, "group 0")
        Self.expectClose(Array(v[32 ..< 34]), [0.5, 1], tol: 0, "group 1")
        #expect(v[6 ..< 32].allSatisfy { $0 == 0 } && v[34 ..< 64].allSatisfy { $0 == 0 })
    }

    @Test("MXFP4 reference agrees with MLX dequantized(mode: .mxfp4) on random bits")
    func mxfp4MatchesMLX() {
        MLXRandom.seed(11)
        let rows = 4, cols = 64
        let bytes = MLXRandom.randInt(low: 0, high: 256, [rows, cols / 2]).asType(.uint8)
        // Keep E8M0 exponents in a sane range (120...134) so values are finite and comparable.
        let scales = (MLXRandom.randInt(low: 120, high: 135, [rows, cols / 32])).asType(.uint8)
        let ref = Ref.mxfp4Dequant(packed: bytes, scales: scales)
        let mlx = dequantized(
            bytes.view(dtype: .uint32), scales: scales, biases: nil, groupSize: 32, bits: 4, mode: .mxfp4,
            dtype: .float32)
        eval(ref, mlx)
        #expect(mlx.shape == ref.shape)
        let diff = abs(ref - mlx).max().item(Float.self)
        #expect(diff == 0, "max |ref - MLX| = \(diff)")
    }

    // Qwen4Exp PLE ---------------------------------------------------------------------------

    @Test("PLE gate hand case (h = 2, hc = 1, eps = 0)")
    func pleGateHandCase() {
        // key = hidden = [1,1], norm weights 0 -> k_n = q_n = [1,1]; d = 2/sqrt(2) = 1.41421356;
        // g = sigmoid(sqrt(d)) = 0.76659923 (python float64); gated = g*[2,4] = [1.53319845, 3.06639691];
        // normed = gated / rms(gated) = [0.63245553, 1.26491106].
        let ones = MLXArray([Float(1), 1], [1, 2])
        let z = MLXArray([Float(0), 0])
        let r = Ref.pleGate(
            key: ones, value: MLXArray([Float(2), 4], [1, 2]), hidden: ones,
            normKey: z, normQuery: z, normConv: z, eps: 0)
        Self.expectClose(Self.values(r.gated), [1.5331985, 3.0663969], tol: 1e-5)
        Self.expectClose(Self.values(r.normed), [0.6324555, 1.2649111], tol: 1e-5)
    }

    @Test("PLE dilated causal short conv (K = 2, dilation = 2): taps t and t-2, zero history, SiLU")
    func pleConvHandCase() {
        // w = [0.5 (tap t-2), 1.0 (tap t)], x = 1,2,3,4 -> pre-activation 1, 2, 3.5, 5 ;
        // silu (python float64): 0.73105858, 1.76159416, 3.39740719, 4.96653575
        let out = Ref.pleDilatedShortConv(
            x: MLXArray([Float(1), 2, 3, 4], [4, 1]), w: MLXArray([Float(0.5), 1.0], [1, 2]), dilation: 2)
        #expect(out.shape == [4, 1])
        Self.expectClose(Self.values(out), [0.7310586, 1.7615942, 3.3974072, 4.9665357], tol: 1e-5)
    }
}
