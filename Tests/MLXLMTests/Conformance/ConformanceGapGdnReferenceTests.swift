// Conformance gap pins, round 3 (SL-29/SL-31): GREEN tests of the gated-delta-rule references in
// GapReference/GdnReference.swift. They pin the spec of the recurrence family so the later kernels have an
// oracle; they test reference code (and, where stated, existing public/internal GatedDelta code) only.
//
// Decisions these tests back (summary in scripts/conformance/README.md; the SL-31 kernel plan is not in the repo):
//   Qwen4Exp GDN       -> composable on GatedDelta.swift: scalar gate, flat [q|k|v|z] layout, silu conv; the
//                         only deltas are the `output_gate_type` (silu|sigmoid) of the output norm and the
//                         spec-decode state handling, neither needs a kernel.
//   Qwen4Exp rollback  -> state after accepting k of n tokens == recomputing k tokens from the checkpoint
//                         (the contract MambaCache.checkpoint/trim and Qwen35 `checkpointAfter` implement).
//   KDA / per-key gate -> composable only through the slow ops path (`gatedDeltaOps` already accepts a
//                         [B,T,Hv,Dk] gate); the Metal kernel has NO per-key-gate variant -> SL-31 kernel.
//   MuseSpark GDN2     -> INFERRED (vLLM raises NotImplementedError); the tests are generic over the gate shape (scalar and per-key), they do not pin one candidate.
// Numeric oracles marked "python float64" were computed once with a pure-python float64 script (no numpy,
// independent of the MLX code); that script is not in the repo (source not in repo). Derivation in the tests:
// S_t = diag(exp(g_t)) S_{t-1}; err = v_t - S_t k_t; S_t += beta_t * outer(err, k_t); y_t = S_t q_t. upstream-watch: Libraries/MLXLMCommon/GatedDelta.swift.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

enum ConformanceGdn {
    typealias Ref = SharpAIOverlayGdnReference

    static func values(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }

    static func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    struct Case {
        var q, k, v, logDecay, beta: MLXArray
    }

    /// Random, deterministic inputs. q/k are L2-normalised (and q scaled) like the real layer feeds them.
    static func makeCase(
        B: Int = 1, T: Int = 12, Hk: Int = 2, Hv: Int = 4, Dk: Int = 8, Dv: Int = 6,
        perKey: Bool, decayScale: Float = 0.5, seed: UInt64 = 7
    ) -> Case {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let (q, k) = Ref.prepQK(q: MLXRandom.normal([B, T, Hk, Dk]), k: MLXRandom.normal([B, T, Hk, Dk]))
            let v = MLXRandom.normal([B, T, Hv, Dv])
            let logDecay = -abs(MLXRandom.normal([B, T, Hv, perKey ? Dk : 1])) * decayScale
            let beta = sigmoid(MLXRandom.normal([B, T, Hv]))
            eval(q, k, v, logDecay, beta)
            return Case(q: q, k: k, v: v, logDecay: logDecay, beta: beta)
        }
    }

    static func randomState(B: Int = 1, Hv: Int = 4, Dv: Int = 6, Dk: Int = 8, seed: UInt64 = 99) -> MLXArray {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            let s = MLXRandom.normal([B, Hv, Dv, Dk])
            eval(s)
            return s
        }
    }

    /// Hand/float64 case: T=3, Hk=1, Hv=2, Dk=Dv=2. Inputs are the literals below (the python script that produced the oracle numbers is not in the repo).
    static func handCase() -> (q: MLXArray, k: MLXArray, v: MLXArray, lg: MLXArray, beta: MLXArray) {
        let q = MLXArray([Float](arrayLiteral: 0.5, 0.75, 1.0, 0.25, -0.5, 0.75), [1, 3, 1, 2])
        let k = MLXArray([Float](arrayLiteral: 1.0, 0.5, -0.5, 1.0, 0.25, -1.0), [1, 3, 1, 2])
        let v = MLXArray(
            [Float](arrayLiteral: 1, 2, -1, 0.5, 0.5, -0.5, 2, 1, -1, 1.5, 0.25, -2), [1, 3, 2, 2])
        let lg = MLXArray(
            [Float](arrayLiteral: -0.1, -0.7, -0.3, -0.3, -0.5, -0.05, -1.0, -0.2, -0.25, -0.4, -0.6, -0.1),
            [1, 3, 2, 2])
        let beta = MLXArray([Float](arrayLiteral: 0.5, 0.25, 0.75, 1.0, 0.5, 0.5), [1, 3, 2])
        return (q, k, v, lg, beta)
    }
}

struct ConformanceGapGdnReferenceTests {
    typealias G = ConformanceGdn
    typealias Ref = SharpAIOverlayGdnReference

    // MARK: hand / float64 oracles

    @Test("KDA recurrence: per-key-dim gate matches the python float64 oracle (y and final state)")
    func kdaHandCase() {
        let c = G.handCase()
        let r = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.lg, beta: c.beta, keepStates: true)
        // python float64 values (generator script not in the repo; recompute by the recurrence in the file header)
        let y: [Float] = [
            0.4375, 0.875, -0.21875, 0.109375, 0.285124923487, 0.851499846974, -0.631644299824, -0.184177850088,
            0.507280233179, -0.923472143763, 0.683282442954, 1.25857421014,
        ]
        let final: [Float] = [
            0.0326532508625, 0.698142478147, 0.832928328214, -0.676010639541, -0.343281102978, 0.68218918862,
            -0.368186366468, 1.43264136921,
        ]
        let after1: [Float] = [0.5, 0.25, 1.0, 0.5, -0.25, -0.125, 0.125, 0.0625]
        ConformanceGapReferenceTests.expectClose(G.values(r.y), y, tol: 1e-5, "kda y")
        ConformanceGapReferenceTests.expectClose(G.values(r.state), final, tol: 1e-5, "kda final state")
        ConformanceGapReferenceTests.expectClose(G.values(r.states![0..., 0]), after1, tol: 1e-6, "kda state after token 0")
        #expect(r.y.shape == [1, 3, 2, 2] && r.state.shape == [1, 2, 2, 2] && r.states!.shape == [1, 3, 2, 2, 2])
    }

    @Test("scalar-gate recurrence matches the python float64 oracle")
    func scalarHandCase() {
        let c = G.handCase()
        // scalar gate = the first key-dim value of each head, shape [..., 1]
        let lg = c.lg[0..., 0..., 0..., 0 ..< 1]
        let r = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: lg, beta: c.beta)
        let y: [Float] = [
            0.4375, 0.875, -0.21875, 0.109375, 0.247423496088, 0.776096992177, -0.603466092829, -0.198266953585,
            0.530451334346, -0.915267700544, 0.459841692009, 1.1477784719,
        ]
        let final: [Float] = [
            0.013608429476, 0.716340732112, 0.779583581795, -0.700634546195, -0.415259809488, 0.33628238302,
            -0.421203458699, 1.24956899007,
        ]
        ConformanceGapReferenceTests.expectClose(G.values(r.y), y, tol: 1e-5, "scalar y")
        ConformanceGapReferenceTests.expectClose(G.values(r.state), final, tol: 1e-5, "scalar final state")
        // A scalar gate IS the per-key gate with equal entries.
        let broadcastLg = broadcast(lg, to: [1, 3, 2, 2])
        let kda = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: broadcastLg, beta: c.beta)
        #expect(G.maxDiff(kda.y, r.y) < 1e-6 && G.maxDiff(kda.state, r.state) < 1e-6)
    }

    @Test("softplus reference is relatively accurate for x << 0 (CI's logAddExp lost it: quantised to ~1e-7)")
    func softplusRelativeAccuracy() {
        // python float64 math.log1p(math.exp(x)) for x < 0, x + log1p(exp(-x)) otherwise.
        let xs: [Float] = [-35, -20, -15.5, -9, -5.85, -2.4, -0.5, 0, 0.7, 3, 25]
        let want: [Double] = [
            6.305116760146987e-16, 2.061153620314381e-09, 1.8553911904921443e-07, 0.00012340218972325883,
            0.0028757601931381368, 0.08683615215394969, 0.4740769841801067, 0.6931471805599453,
            1.103186048885458, 3.048587351573742, 25.000000000013888,
        ]
        let got = G.values(Ref.softplus(MLXArray(xs)))
        #expect(got.count == want.count)
        for (i, (g, w)) in zip(got, want).enumerated() {
            // The float32 input x is exact, so only the output rounding (u = 6e-8) and the exp of a float32-rounded x remain.
            let relative = abs(Double(g) - w) / w
            #expect(relative <= 2e-6, "softplus(\(xs[i])): got \(g), want \(w), relative error \(relative)")
        }
    }

    @Test("gate maths: softplus gate, KDA lower-bound safe gate, beta (and allow_neg_eigval), softplus threshold")
    func gateMaths() {
        let aLog = MLXArray([Float(0), Float(log(2.0))])
        // scalar GDN gate, a=[0.5,-1], dtBias=[0.25,0]: python float64 -> [-1.13687100611, -0.626523375036]
        let a = MLXArray([Float(0.5), -1], [1, 1, 2])
        let sc = Ref.scalarLogDecay(aLog: aLog, a: a, dtBias: MLXArray([Float(0.25), 0]))
        #expect(sc.shape == [1, 1, 2, 1])
        ConformanceGapReferenceTests.expectClose(G.values(sc), [-1.13687100611, -0.626523375036], tol: 1e-5)
        // KDA gate on the same numbers (Dk = 1 per head)
        let g = MLXArray([Float(0.5), -1], [1, 1, 2, 1]), bias = MLXArray([Float(0.25), 0], [2, 1])
        ConformanceGapReferenceTests.expectClose(
            G.values(Ref.kdaLogDecay(aLog: aLog, g: g, bias: bias)), [-1.13687100611, -0.626523375036], tol: 1e-5)
        // safe gate, lower bound -5: python float64 -> [-3.39589349588, -0.596014610111]
        ConformanceGapReferenceTests.expectClose(
            G.values(Ref.kdaLogDecay(aLog: aLog, g: g, bias: bias, lowerBound: -5)),
            [-3.39589349588, -0.596014610111], tol: 1e-5)
        // the decay exp(logDecay) is always in (0, 1]; the safe gate never goes below the bound
        let big = MLXArray([Float(-50), 0, 50], [1, 1, 3, 1])
        let bigBias = MLXArray.zeros([3, 1])
        let sp = Ref.kdaLogDecay(aLog: MLXArray.zeros([3]), g: big, bias: bigBias)
        ConformanceGapReferenceTests.expectClose(G.values(sp), [0, -0.6931472, -50], tol: 1e-4)  // softplus(50)=50 (no overflow)
        let safe = Ref.kdaLogDecay(aLog: MLXArray.zeros([3]), g: big, bias: bigBias, lowerBound: -5)
        ConformanceGapReferenceTests.expectClose(G.values(safe), [0, -2.5, -5], tol: 1e-4)
        // beta: sigmoid(0)=0.5; allow_neg_eigval doubles it (OlmoHybrid, range (0,2))
        ConformanceGapReferenceTests.expectClose(G.values(Ref.beta(MLXArray([Float(0)]))), [0.5])
        ConformanceGapReferenceTests.expectClose(G.values(Ref.beta(MLXArray([Float(0)]), allowNegEigval: true)), [1.0])
    }

    @Test("q/k prep: L2 norm with eps inside the sqrt (1e-6 added to the sum of squares) and q * Dk^-0.5")
    func prepQKHand() {
        // x = [3,4] -> /sqrt(25+1e-6) ; q additionally * 2^-0.5 (Dk = 2)
        let (q, k) = Ref.prepQK(q: MLXArray([Float(3), 4], [1, 1, 1, 2]), k: MLXArray([Float(3), 4], [1, 1, 1, 2]))
        let n = Float(25 + 1e-6).squareRoot()
        ConformanceGapReferenceTests.expectClose(G.values(k), [3 / n, 4 / n], tol: 1e-6)
        ConformanceGapReferenceTests.expectClose(G.values(q), [3 / n / Float(2).squareRoot(), 4 / n / Float(2).squareRoot()], tol: 1e-6)
    }

    // MARK: chunked == recurrent

    @Test("property: chunked delta rule == recurrent for scalar and per-key gates, any chunk size, GQA, non-zero state")
    func chunkedEqualsRecurrent() {
        var worst: Float = 0
        for perKey in [false, true] {
            for chunk in [1, 2, 3, 5, 12, 64] {  // 64 > T: single partial chunk; 5: ragged last chunk
                for withState in [false, true] {
                    let c = G.makeCase(perKey: perKey, seed: UInt64(10 + chunk))
                    let s0 = withState ? G.randomState() : nil
                    let want = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, state: s0)
                    let got = Ref.chunked(
                        q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, state: s0, chunk: chunk)
                    let dy = G.maxDiff(got.y, want.y), ds = G.maxDiff(got.state, want.state)
                    worst = max(worst, dy, ds)
                    #expect(
                        dy < 2e-4 && ds < 2e-4,
                        "perKey=\(perKey) chunk=\(chunk) state=\(withState): |dy|=\(dy) |dstate|=\(ds)")
                }
            }
        }
        print("[gdn] chunked-vs-recurrent worst abs diff = \(worst)")
    }

    @Test("chunked == recurrent under very strong decay (log-decay down to -40): pairwise exp(G_t - G_s) never overflows")
    func chunkedStrongDecay() {
        let c = G.makeCase(T: 16, perKey: true, decayScale: 40, seed: 3)
        let want = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta)
        let got = Ref.chunked(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, chunk: 8)
        #expect(G.values(got.y).allSatisfy { $0.isFinite } && G.values(got.state).allSatisfy { $0.isFinite })
        #expect(G.maxDiff(got.y, want.y) < 2e-4 && G.maxDiff(got.state, want.state) < 2e-4)
    }

    @Test("chunk-by-chunk carry: running the chunked form on two halves, passing the state, == one run")
    func chunkedCarry() {
        let c = G.makeCase(T: 12, perKey: true, seed: 21)
        let full = Ref.chunked(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, chunk: 4)
        func part(_ r: Range<Int>) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
            (c.q[0..., r], c.k[0..., r], c.v[0..., r], c.logDecay[0..., r], c.beta[0..., r])
        }
        let a = part(0 ..< 5), b = part(5 ..< 12)
        let first = Ref.chunked(q: a.0, k: a.1, v: a.2, logDecay: a.3, beta: a.4, chunk: 4)
        let second = Ref.chunked(q: b.0, k: b.1, v: b.2, logDecay: b.3, beta: b.4, state: first.state, chunk: 4)
        #expect(G.maxDiff(concatenated([first.y, second.y], axis: 1), full.y) < 2e-4)
        #expect(G.maxDiff(second.state, full.state) < 2e-4)
    }

    // MARK: composability on the existing GatedDelta.swift

    @Test("Qwen4Exp/Qwen3.5 scalar-gate GDN is composable: reference == public gatedDeltaUpdate (kernel and ops paths)")
    func scalarGateMatchesGatedDeltaUpdate() {
        let (B, T, Hk, Hv, Dk, Dv) = (1, 10, 2, 4, 32, 16)  // Dk % 32 == 0 -> the Metal kernel path
        let inp = withRandomState(MLXRandom.RandomState(seed: 5)) {
            () -> [MLXArray] in
            let (q, k) = Ref.prepQK(q: MLXRandom.normal([B, T, Hk, Dk]), k: MLXRandom.normal([B, T, Hk, Dk]))
            let arrs = [
                q, k, MLXRandom.normal([B, T, Hv, Dv]), MLXRandom.normal([B, T, Hv]), MLXRandom.normal([B, T, Hv]),
                MLXRandom.normal([Hv]) * 0.1, MLXRandom.normal([Hv]),
            ]
            eval(arrs)
            return arrs
        }
        let want = Ref.recurrent(
            q: inp[0], k: inp[1], v: inp[2], logDecay: Ref.scalarLogDecay(aLog: inp[5], a: inp[3], dtBias: inp[6]),
            beta: Ref.beta(inp[4]))
        for useKernel in [true, false] {
            let got = gatedDeltaUpdate(
                q: inp[0], k: inp[1], v: inp[2], a: inp[3], b: inp[4], aLog: inp[5], dtBias: inp[6],
                useKernel: useKernel)
            eval(got.0, got.1)
            #expect(G.maxDiff(got.0, want.y) < 1e-3, "useKernel=\(useKernel) y diff \(G.maxDiff(got.0, want.y))")
            #expect(G.maxDiff(got.1, want.state) < 1e-3, "useKernel=\(useKernel) state diff \(G.maxDiff(got.1, want.state))")
        }
    }

    @Test("KDA is composable on the OPS path: gatedDeltaOps already takes a [B,T,Hv,Dk] gate and equals the reference")
    func perKeyGateOpsPath() {
        let c = G.makeCase(T: 9, Hk: 2, Hv: 4, Dk: 32, Dv: 16, perKey: true, seed: 8)
        let want = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta)
        let got = gatedDeltaOps(q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta)
        eval(got.0, got.1)
        #expect(G.maxDiff(got.0, want.y) < 1e-4 && G.maxDiff(got.1, want.state) < 1e-4)
        // and the scalar-gate ops path agrees with a [B,T,Hv,1] reference gate expanded to Dk
        let s = G.makeCase(T: 9, Hk: 2, Hv: 4, Dk: 32, Dv: 16, perKey: false, seed: 9)
        let w2 = Ref.recurrent(q: s.q, k: s.k, v: s.v, logDecay: s.logDecay, beta: s.beta)
        let g2 = gatedDeltaOps(q: s.q, k: s.k, v: s.v, g: exp(s.logDecay[0..., 0..., 0..., 0]), beta: s.beta)
        eval(g2.0)
        #expect(G.maxDiff(g2.0, w2.y) < 1e-4)
    }

    // MARK: Qwen4Exp deltas

    @Test("Qwen4Exp output gate: RMSNorm * silu(z) (swish) vs * sigmoid(z) -- hand values")
    func outputGateHand() {
        // x = [1,1] with weight 1, eps 0 -> rms-normed x = [1,1]. z = 0 and 2.
        let x = MLXArray([Float(1), 1], [1, 2]), w = MLXArray([Float(1), 1])
        let z = MLXArray([Float(0), 2], [1, 2])
        // silu(2) = 2*sigmoid(2) = 1.7615941559557646 ; sigmoid(2) = 0.8807970779778823 ; sigmoid(0)=0.5, silu(0)=0
        ConformanceGapReferenceTests.expectClose(
            G.values(Ref.gatedRMSNorm(x, z: z, weight: w, eps: 0, kind: "silu")), [0, 1.7615942], tol: 1e-6)
        ConformanceGapReferenceTests.expectClose(
            G.values(Ref.gatedRMSNorm(x, z: z, weight: w, eps: 0, kind: "swish")), [0, 1.7615942], tol: 1e-6)
        ConformanceGapReferenceTests.expectClose(
            G.values(Ref.gatedRMSNorm(x, z: z, weight: w, eps: 0, kind: "sigmoid")), [0.5, 0.8807971], tol: 1e-6)
    }

    @Test("existing Qwen3NextRMSNormGated (used by Qwen3.5 GDN) == the silu output gate")
    func existingNormIsSilu() {
        let (x, z) = withRandomState(MLXRandom.RandomState(seed: 4)) {
            () -> (MLXArray, MLXArray) in
            let r = (MLXRandom.normal([2, 3, 4, 16]), MLXRandom.normal([2, 3, 4, 16]))
            eval(r.0, r.1)
            return r
        }
        let norm = Qwen3NextRMSNormGated(dimensions: 16, eps: 1e-6)
        let got = norm(x, gate: z)
        let want = Ref.gatedRMSNorm(x, z: z, weight: MLXArray.ones([16]), eps: 1e-6, kind: "silu")
        eval(got, want)
        #expect(G.maxDiff(got, want) < 1e-4)
        // and it is NOT the sigmoid gate: the delta Qwen4Exp (output_gate_type = sigmoid) needs is real
        let sig = Ref.gatedRMSNorm(x, z: z, weight: MLXArray.ones([16]), eps: 1e-6, kind: "sigmoid")
        #expect(G.maxDiff(got, sig) > 1e-2)
    }

    @Test("Qwen4Exp flat layout: qkvz = [q|k|v|z], ba = [b|a] (Qwen3.5 layout; Qwen3-Next interleaves per key head)")
    func flatSplitLayout() {
        // key_dim 2 (1 key head x 2), value_dim 4 (2 value heads x 2): qkvz = 12 columns
        let qkvz = MLXArray(Array(0 ..< 12).map { Float($0) }, [1, 12])
        let ba = MLXArray([Float(100), 101, 200, 201], [1, 4])  // b = [100,101], a = [200,201]
        let s = Ref.splitFlatQKVZBA(qkvz: qkvz, ba: ba, keyDim: 2, valueDim: 4, numVHeads: 2)
        #expect(G.values(s.q) == [0, 1] && G.values(s.k) == [2, 3])
        #expect(G.values(s.v) == [4, 5, 6, 7] && G.values(s.z) == [8, 9, 10, 11])
        #expect(G.values(s.b) == [100, 101] && G.values(s.a) == [200, 201])
    }

    // MARK: decode-state carry and speculative rollback

    @Test("rollback: state after accepting k of n speculative tokens == recomputing k tokens, for every k (0...n)")
    func rollbackEqualsRecompute() {
        for perKey in [false, true] {
            let n = 6
            let c = G.makeCase(T: n, perKey: perKey, seed: 31)
            let s0 = G.randomState(seed: 32)
            let verify = Ref.recurrent(
                q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, state: s0, keepStates: true)
            for k in 0 ... n {
                let rolled = Ref.stateAfterAccepting(k, states: verify.states!, initial: s0)
                let recomputed: MLXArray
                if k == 0 {
                    recomputed = s0
                } else {
                    // what the checkpoint + re-run path computes: restore s0, run the first k tokens again
                    recomputed = Ref.recurrent(
                        q: c.q[0..., ..<k], k: c.k[0..., ..<k], v: c.v[0..., ..<k],
                        logDecay: c.logDecay[0..., ..<k], beta: c.beta[0..., ..<k], state: s0).state
                }
                #expect(G.maxDiff(rolled, recomputed) < 1e-6, "perKey=\(perKey) k=\(k)")
            }
            #expect(G.maxDiff(verify.states![0..., n - 1], verify.state) == 0)
        }
    }

    @Test("gatedDeltaUpdate over the first k tokens == the reference state after token k-1 (rollback target); the n-token state is not it. No MambaCache involved")
    func rollbackThroughGatedDeltaUpdate() {
        let (B, n, Hk, Hv, Dk, Dv) = (1, 5, 2, 4, 32, 16)
        let inp = withRandomState(MLXRandom.RandomState(seed: 41)) {
            () -> [MLXArray] in
            let (q, k) = Ref.prepQK(q: MLXRandom.normal([B, n, Hk, Dk]), k: MLXRandom.normal([B, n, Hk, Dk]))
            let arrs = [
                q, k, MLXRandom.normal([B, n, Hv, Dv]), MLXRandom.normal([B, n, Hv]), MLXRandom.normal([B, n, Hv]),
                MLXRandom.normal([Hv]) * 0.1, MLXRandom.normal([Hv]), MLXRandom.normal([B, Hv, Dv, Dk]),
            ]
            eval(arrs)
            return arrs
        }
        let ref = Ref.recurrent(
            q: inp[0], k: inp[1], v: inp[2], logDecay: Ref.scalarLogDecay(aLog: inp[5], a: inp[3], dtBias: inp[6]),
            beta: Ref.beta(inp[4]), state: inp[7], keepStates: true)
        let full = gatedDeltaUpdate(
            q: inp[0], k: inp[1], v: inp[2], a: inp[3], b: inp[4], aLog: inp[5], dtBias: inp[6], state: inp[7])
        for k in 1 ..< n {
            let part = gatedDeltaUpdate(
                q: inp[0][0..., ..<k], k: inp[1][0..., ..<k], v: inp[2][0..., ..<k],
                a: inp[3][0..., ..<k], b: inp[4][0..., ..<k], aLog: inp[5], dtBias: inp[6], state: inp[7])
            eval(part.1)
            #expect(G.maxDiff(part.1, ref.states![0..., k - 1]) < 1e-3, "k=\(k)")
            // negative control: keeping the n-token state (no rollback) is NOT the k-token state
            #expect(G.maxDiff(full.1, ref.states![0..., k - 1]) > 1e-2, "k=\(k): rollback must matter")
        }
    }

    @Test("rollback then continue: accept k, resume with the real next tokens == never having speculated")
    func rollbackThenContinue() {
        let n = 5, m = 4
        let all = G.makeCase(T: n + m, perKey: true, seed: 51)
        let c = G.makeCase(T: n, perKey: true, seed: 52)  // the REJECTED drafts' inputs (different from `all`)
        let s0 = G.randomState(seed: 53)
        for k in 0 ... n {
            // Path A: verify n draft tokens (junk past k), roll back to k, then process the real continuation.
            let verifyQ = concatenated([all.q[0..., ..<k], c.q[0..., k...]], axis: 1)
            let verifyK = concatenated([all.k[0..., ..<k], c.k[0..., k...]], axis: 1)
            let verifyV = concatenated([all.v[0..., ..<k], c.v[0..., k...]], axis: 1)
            let verifyG = concatenated([all.logDecay[0..., ..<k], c.logDecay[0..., k...]], axis: 1)
            let verifyB = concatenated([all.beta[0..., ..<k], c.beta[0..., k...]], axis: 1)
            let v = Ref.recurrent(q: verifyQ, k: verifyK, v: verifyV, logDecay: verifyG, beta: verifyB, state: s0, keepStates: true)
            let rolled = Ref.stateAfterAccepting(k, states: v.states!, initial: s0)
            let rest = k ..< (n + m)
            let a = Ref.recurrent(
                q: all.q[0..., rest], k: all.k[0..., rest], v: all.v[0..., rest],
                logDecay: all.logDecay[0..., rest], beta: all.beta[0..., rest], state: rolled)
            // Path B: never speculated.
            let b = Ref.recurrent(q: all.q, k: all.k, v: all.v, logDecay: all.logDecay, beta: all.beta, state: s0)
            #expect(G.maxDiff(a.state, b.state) < 1e-5, "k=\(k)")
            #expect(G.maxDiff(a.y, b.y[0..., rest]) < 1e-5, "k=\(k)")
        }
    }

    @Test("chunk-boundary state == per-token slot: a chunked prefill can only roll back to chunk edges")
    func chunkBoundaryState() {
        let c = G.makeCase(T: 12, perKey: true, seed: 61)
        let per = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, keepStates: true)
        for k in [4, 8, 12] {
            let part = Ref.chunked(
                q: c.q[0..., ..<k], k: c.k[0..., ..<k], v: c.v[0..., ..<k], logDecay: c.logDecay[0..., ..<k],
                beta: c.beta[0..., ..<k], chunk: 4)
            #expect(G.maxDiff(part.state, per.states![0..., k - 1]) < 2e-4, "k=\(k)")
        }
    }

    @Test("conv-state rollback: last K-1 rows of [prev ++ x[:k]] -- hand case and == k single-token updates")
    func convStateRollback() {
        // K = 4 -> 3 rows of history, C = 1. prev = [1,2,3], x = [10,20,30,40]
        let prev = MLXArray([Float(1), 2, 3], [1, 3, 1]), x = MLXArray([Float(10), 20, 30, 40], [1, 4, 1])
        let want: [Int: [Float]] = [0: [1, 2, 3], 1: [2, 3, 10], 2: [3, 10, 20], 3: [10, 20, 30], 4: [20, 30, 40]]
        for (k, rows) in want {
            #expect(G.values(Ref.convStateAfterAccepting(k, prev: prev, x: x)) == rows, "k=\(k)")
        }
        // == feeding the k tokens one at a time (the decode path: state = concat(state[1:], token))
        var state = prev
        for i in 0 ..< 4 {
            state = concatenated([state[0..., 1...], x[0..., i ..< (i + 1)]], axis: 1)
            #expect(G.values(Ref.convStateAfterAccepting(i + 1, prev: prev, x: x)) == G.values(state))
        }
    }

    @Test("characterization: MambaCache.trim(m>0) restores the WHOLE pre-draft checkpoint (no partial trim); accepted tokens are recomputed")
    func mambaCacheTrimIsAllOrNothing() {
        let cache = MambaCache()
        cache[0] = MLXArray.zeros([1, 3, 4]); cache[1] = MLXArray.zeros([1, 2, 3, 4])
        cache.checkpoint()
        cache[1] = MLXArray.ones([1, 2, 3, 4])  // state after a 5-token draft round
        // Reject 2 of 5 drafts: a per-token-slot design would leave the 3-token state. trim() reports 2 but
        // the cache holds the pre-draft checkpoint (all 5 rolled back) -- the caller re-runs the accepted 3.
        let trimmed = cache.trim(2)
        #expect(trimmed == 2)
        #expect(cache[1]!.sum().item(Float.self) == 0, "state is the checkpoint, not a partially-rolled state")
        // With nothing rejected the checkpoint is dropped and the (full) state kept.
        cache.checkpoint(); cache[1] = MLXArray.ones([1, 2, 3, 4])
        #expect(cache.trim(0) == 0 && cache[1]!.sum().item(Float.self) == 24)
    }
}
