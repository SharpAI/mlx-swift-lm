// The production decay gate of gatedDeltaUpdate (GatedDelta.swift: exp(-exp(aLog) * softplus(a + dtBias)),
// `computeGatedDeltaG`, private) observed through the public API, on whatever MLX backend runs the test.
//
// Why this exists: on the CI runner `logAddExp(x, 0)` (MLXNN `softplus`) was found to lose the low bits of e^x for
// x << 0 (PR #94, KDA gate goldens: absolute error 4e-8..7e-8 at values ~1e-7). The golden suite checks the log-decay
// with a relative tolerance, which the shipping code never needs: the model consumes exp(logDecay), where that
// error is an absolute ~exp(aLog) * 6e-8. This test states that consequence on the real function.
//
// How the gate is read back: with beta = sigmoid(-60) ~ 9e-27, v = 0 and k = 0 the delta update is zero, so after
// one step the state is exactly g * state0. state0 = 1 everywhere returns g per value head.
//
// Expected values: python float64 from the float32 inputs below,
//   g = exp(-exp(aLog) * (max(x, 0) + log1p(exp(-|x|)))), x = float32(a) + float32(dtBias).
//
// Tolerance 5e-7 (absolute): output rounding of a value in (0, 1] <= 6e-8, softplus absolute error observed on CI
// <= 7e-8 times exp(aLog) <= 3.3 (aLog <= 1.2) = 2.3e-7, plus the float32 exp, plus ~40% margin.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite("GatedDelta decay gate (production)")
struct ConformanceGatedDeltaGateTests {
    static let aLog: [Float] = [0, 0.5, -1, 1.2, 0, 0, 0.3, 0, -0.5, 0]
    static let dtBias: [Float] = [0, 0, 0.25, 0, -0.5, 0, 0.125, 0, 0, 1.5]
    // a + dtBias = x, in float32: -35, -20, -16 + 0.25.., -9, -5.85, -2.4, -0.5, 0, 0.7, 3
    static let a: [Float] = [-35, -20, -15.75, -9, -5.349999904632568, -2.4000000953674316, -0.625, 0, 0.699999988079071, 1.5]
    static let want: [Double] = [
        0.9999999999999993, 0.9999999966017322, 0.999999931743975, 0.9995903742016258, 0.9971283705712262,
        0.9168273107783208, 0.5273256444103019, 0.5, 0.512161047663063, 0.04742587317756678,
    ]

    @Test("gatedDeltaUpdate applies exp(-exp(aLog) * softplus(a + dtBias)) to the state", arguments: [true, false])
    func gate(useKernel: Bool) {
        let H = Self.aLog.count
        let Dk = 32, Dv = 2
        let q = MLXArray.ones([1, 1, H, Dk])
        let k = MLXArray.zeros([1, 1, H, Dk])
        let v = MLXArray.zeros([1, 1, H, Dv])
        let b = MLXArray([Float](repeating: -60, count: H), [1, 1, H])
        let state0 = MLXArray.ones([1, H, Dv, Dk])
        let (_, state) = gatedDeltaUpdate(
            q: q, k: k, v: v,
            a: MLXArray(Self.a, [1, 1, H]), b: b,
            aLog: MLXArray(Self.aLog), dtBias: MLXArray(Self.dtBias),
            state: state0, useKernel: useKernel)
        eval(state)
        #expect(state.shape == [1, H, Dv, Dk])
        let got = state.asArray(Float.self)
        for h in 0 ..< H {
            var worst = Double(0)
            for i in 0 ..< (Dv * Dk) { worst = max(worst, abs(Double(got[h * Dv * Dk + i]) - Self.want[h])) }
            #expect(
                worst <= 5e-7,
                "head \(h) (a=\(Self.a[h]), aLog=\(Self.aLog[h]), dtBias=\(Self.dtBias[h])): max|state - g| = \(worst), g = \(Self.want[h]) [kernel: \(useKernel)]"
            )
        }
    }
}
