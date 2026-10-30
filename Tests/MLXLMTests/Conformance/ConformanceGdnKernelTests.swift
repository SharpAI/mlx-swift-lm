// SL-31 K1: numeric comparison of SharpAIOverlayGatedDeltaKDA (Metal kernel `sharpai_gated_delta_kda_step`,
// per-key-dimension gate) against the pure-MLX oracle SharpAIOverlayGdnReference.recurrent.
// Shapes are tiny on purpose (Metal hangs were seen on CI); one GPU eval per case.
// The decay multiplier g = exp(logDecay) is what the kernel takes; the oracle takes logDecay.
// upstream-watch: Libraries/MLXLMCommon/GatedDelta.swift (the kernel is a per-key-gate copy of its structure).

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("ConformanceGdnKernelTests")
struct ConformanceGdnKernelTests {
    typealias G = ConformanceGdn
    typealias Ref = SharpAIOverlayGdnReference
    typealias KDA = SharpAIOverlayGatedDeltaKDA

    struct Shape: CustomStringConvertible {
        var B: Int = 1
        var T: Int
        var Hk: Int
        var Hv: Int
        var Dk: Int
        var Dv: Int = 8
        var withState: Bool = false
        var masked: Bool = false
        var description: String { "B\(B) T\(T) Hk\(Hk) Hv\(Hv) Dk\(Dk) Dv\(Dv) state=\(withState) mask=\(masked)" }
    }

    /// Runs kernel and oracle on the same (dtype-rounded) inputs; returns (|dy|, |dstate|).
    static func compare(_ s: Shape, perKey: Bool, dtype: DType, seed: UInt64 = 5, useKernel: Bool = true) -> (Float, Float) {
        let c = G.makeCase(B: s.B, T: s.T, Hk: s.Hk, Hv: s.Hv, Dk: s.Dk, Dv: s.Dv, perKey: perKey, seed: seed)
        let q = c.q.asType(dtype), k = c.k.asType(dtype), v = c.v.asType(dtype)
        let state: MLXArray? = s.withState ? G.randomState(B: s.B, Hv: s.Hv, Dv: s.Dv, Dk: s.Dk, seed: seed + 1) : nil
        var logDecay = broadcast(c.logDecay, to: [s.B, s.T, s.Hv, s.Dk])
        var beta = c.beta
        var mask: MLXArray? = nil
        if s.masked {
            // mask out every other step (and step 0 when T > 1)
            let m = (0 ..< s.B * s.T).map { ($0 % s.T) % 2 == 1 || s.T == 1 }
            mask = MLXArray(m).reshaped(s.B, s.T)
        }
        // Oracle for a masked step: identity on the state (decay 1, beta 0); y is forced to 0 below.
        var want: (y: MLXArray, state: MLXArray, states: MLXArray?)
        if let mask {
            let m4 = mask.reshaped(s.B, s.T, 1, 1)
            logDecay = MLX.where(m4, logDecay, MLXArray(Float(0)))
            beta = MLX.where(mask.reshaped(s.B, s.T, 1), beta, MLXArray(Float(0)))
            want = Ref.recurrent(q: q, k: k, v: v, logDecay: logDecay, beta: beta, state: state)
            want.y = MLX.where(mask.reshaped(s.B, s.T, 1, 1), want.y, MLXArray(Float(0)))
        } else {
            want = Ref.recurrent(q: q, k: k, v: v, logDecay: logDecay, beta: beta, state: state)
        }
        // Kernel gets the ORIGINAL (unmodified) gate / beta and the mask.
        let g = exp(broadcast(c.logDecay, to: [s.B, s.T, s.Hv, s.Dk]))
        let got = KDA.update(q: q, k: k, v: v, g: g, beta: c.beta, state: state, mask: mask, useKernel: useKernel)
        eval(got.y, got.state, want.y, want.state)
        #expect(got.y.dtype == dtype)
        #expect(got.state.dtype == .float32)
        return (G.maxDiff(got.y, want.y), G.maxDiff(got.state, want.state))
    }

    static let shapes: [Shape] = [
        Shape(T: 1, Hk: 2, Hv: 4, Dk: 32),
        Shape(T: 1, Hk: 4, Hv: 4, Dk: 32, withState: true),
        Shape(T: 7, Hk: 2, Hv: 4, Dk: 32, withState: true),
        Shape(T: 7, Hk: 4, Hv: 4, Dk: 128),
        Shape(B: 2, T: 5, Hk: 2, Hv: 4, Dk: 64, withState: true),
        Shape(T: 9, Hk: 2, Hv: 4, Dk: 32, withState: true, masked: true),
        Shape(T: 1, Hk: 2, Hv: 4, Dk: 32, withState: true, masked: true),
        Shape(T: 24, Hk: 2, Hv: 4, Dk: 32),
    ]

    @Test("kernel == reference, float32, per-key and broadcast gate", arguments: [true, false])
    func float32(perKey: Bool) {
        for s in Self.shapes {
            let (dy, ds) = Self.compare(s, perKey: perKey, dtype: .float32)
            #expect(dy < 1e-4 && ds < 1e-4, "\(s) perKey=\(perKey): |dy|=\(dy) |dstate|=\(ds)")
        }
    }

    @Test("kernel == reference, float16 and bfloat16 inputs (fp32 state)", arguments: [DType.float16, DType.bfloat16])
    func lowPrecision(dtype: DType) {
        let ytol: Float = dtype == .bfloat16 ? 3e-2 : 1e-2
        for s in Self.shapes {
            let (dy, ds) = Self.compare(s, perKey: true, dtype: dtype)
            #expect(dy < ytol && ds < 2e-3, "\(s) \(dtype): |dy|=\(dy) |dstate|=\(ds)")
        }
    }

    @Test("a per-key gate really differs from a scalar one on the same data (discriminating case)")
    func discriminating() {
        let c = G.makeCase(T: 8, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: true, seed: 71)
        let a = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta)
        let b = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay.mean(axis: -1, keepDims: true), beta: c.beta)
        eval(a.y, b.y)
        #expect(G.maxDiff(a.y, b.y) > 1e-2)
    }

    @Test("broadcast gate equals the existing scalar-gate kernel (gatedDeltaKernel), 1e-5")
    func scalarEquivalence() {
        let c = G.makeCase(T: 6, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: false, seed: 13)
        let gS = exp(c.logDecay).reshaped(1, 6, 4)  // [B,T,Hv]
        let zero = MLXArray.zeros([1, 4, 8, 32])
        let old = gatedDeltaKernel(q: c.q, k: c.k, v: c.v, g: gS, beta: c.beta, state: zero)
        let new = KDA.update(
            q: c.q, k: c.k, v: c.v, g: broadcast(exp(c.logDecay), to: [1, 6, 4, 32]), beta: c.beta, state: zero)
        eval(old.0, old.1, new.y, new.state)
        #expect(G.maxDiff(old.0, new.y) < 1e-5 && G.maxDiff(old.1, new.state) < 1e-5)
    }

    @Test("edge gates: g == 0 (full forget) and g == 1 (no decay)")
    func edgeGates() {
        let c = G.makeCase(T: 5, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: true, seed: 3)
        for value in [Float(0), 1] {
            let g = MLXArray.ones([1, 5, 4, 32]) * value
            let want = Ref.recurrent(
                q: c.q, k: c.k, v: c.v, logDecay: log(MLX.maximum(g, MLXArray(Float(1e-30)))), beta: c.beta)
            let got = KDA.update(q: c.q, k: c.k, v: c.v, g: g, beta: c.beta)
            eval(got.y, got.state, want.y, want.state)
            #expect(G.maxDiff(got.y, want.y) < 1e-4 && G.maxDiff(got.state, want.state) < 1e-4, "g=\(value)")
        }
    }

    @Test("fallbacks: Dk not a multiple of 32, useKernel false, and T == 0 all agree with the reference")
    func fallbacks() {
        let odd = Self.compare(Shape(T: 4, Hk: 2, Hv: 4, Dk: 16, withState: true), perKey: true, dtype: .float32)
        #expect(odd.0 < 1e-4 && odd.1 < 1e-4, "Dk=16 ops fallback: \(odd)")
        let off = Self.compare(Shape(T: 4, Hk: 2, Hv: 4, Dk: 32, withState: true), perKey: true, dtype: .float32, useKernel: false)
        #expect(off.0 < 1e-4 && off.1 < 1e-4, "useKernel=false: \(off)")

        let st = G.randomState(B: 1, Hv: 4, Dv: 8, Dk: 32, seed: 4)
        let e = KDA.update(
            q: MLXArray.zeros([1, 0, 2, 32]), k: MLXArray.zeros([1, 0, 2, 32]), v: MLXArray.zeros([1, 0, 4, 8]),
            g: MLXArray.zeros([1, 0, 4, 32]), beta: MLXArray.zeros([1, 0, 4]), state: st)
        eval(e.y, e.state)
        #expect(e.y.shape == [1, 0, 4, 8])
        #expect(G.maxDiff(e.state, st) == 0)
    }

    @Test("masked step leaves the state bit-identical and writes y = 0")
    func maskedStepIsIdentity() {
        let c = G.makeCase(T: 4, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: true, seed: 8)
        let st = G.randomState(B: 1, Hv: 4, Dv: 8, Dk: 32, seed: 9)
        let mask = MLXArray([false, false, false, false]).reshaped(1, 4)
        let r = KDA.update(q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta, state: st, mask: mask)
        eval(r.y, r.state)
        #expect(G.maxDiff(r.state, st) == 0)
        #expect(abs(r.y).max().item(Float.self) == 0)
    }

    @Test("masked step: the op-composition fallback also writes y = 0 and keeps the state")
    func maskedStepFallback() {
        let c = G.makeCase(T: 4, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: true, seed: 8)
        let st = G.randomState(B: 1, Hv: 4, Dv: 8, Dk: 32, seed: 9)
        let mask = MLXArray([false, true, false, false]).reshaped(1, 4)
        let r = KDA.update(
            q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta, state: st, mask: mask, useKernel: false)
        let k = KDA.update(q: c.q, k: c.k, v: c.v, g: exp(c.logDecay), beta: c.beta, state: st, mask: mask)
        eval(r.y, r.state, k.y, k.state)
        #expect(abs(r.y[0..., 0]).max().item(Float.self) == 0, "masked step 0 must give y = 0")
        #expect(G.maxDiff(r.y, k.y) < 1e-4 && G.maxDiff(r.state, k.state) < 1e-4)
    }

    @Test("rollback: state after the first k tokens == reference slot k-1 (kernel run on the prefix)")
    func rollbackPrefix() {
        let c = G.makeCase(T: 6, Hk: 2, Hv: 4, Dk: 32, Dv: 8, perKey: true, seed: 21)
        let want = Ref.recurrent(q: c.q, k: c.k, v: c.v, logDecay: c.logDecay, beta: c.beta, keepStates: true)
        let n = 3
        let r = KDA.update(
            q: c.q[0..., ..<n], k: c.k[0..., ..<n], v: c.v[0..., ..<n],
            g: exp(c.logDecay)[0..., ..<n], beta: c.beta[0..., ..<n])
        let slot = Ref.stateAfterAccepting(n, states: want.states!, initial: MLXArray.zeros(r.state.shape))
        eval(r.state, slot)
        #expect(G.maxDiff(r.state, slot) < 1e-4)
    }

    @Test("every metalKernel name under SharpAIOverlay/ carries the sharpai_ prefix")
    func kernelNamePrefix() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Libraries/MLXLMCommon/SharpAIOverlay")
        let files = try FileManager.default.contentsOfDirectory(atPath: root.path).filter { $0.hasSuffix(".swift") }
        var names = [String]()
        for f in files {
            let text = try String(contentsOf: root.appendingPathComponent(f), encoding: .utf8)
            for line in text.split(separator: "\n") where line.contains("name: \"") && !line.contains("//") {
                if let r = line.range(of: "name: \""), let e = line[r.upperBound...].firstIndex(of: "\"") {
                    names.append(String(line[r.upperBound ..< e]))
                }
            }
        }
        #expect(names.contains { $0.hasPrefix("sharpai_gated_delta_kda_step") }, "kernel names found: \(names)")
        for n in names { #expect(n.hasPrefix("sharpai_"), "\(n)") }
    }
}
