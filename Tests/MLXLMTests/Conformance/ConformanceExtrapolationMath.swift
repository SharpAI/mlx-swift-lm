// Conformance harness, SL-29 phase 3: slope extrapolation of speed (pure maths, no MLX).
//
// Method: time a step at a few reduced layer counts, fit  time = a + b * layers  by least squares,
// and evaluate the line at the real layer count. Everything here is deterministic plain Swift so it
// can be unit-tested with hand-made numbers; the (noisy) timing lives in
// ConformanceExtrapolationTimingTests.swift.
//
// upstream-watch: nothing here touches upstream types. If upstream ever adds a type named
// `SharpAIOverlayExtrapolation`, rename ours.

import Foundation

enum SharpAIOverlayExtrapolation {

    struct Fit: Equatable {
        /// time at zero layers (embedding + lm_head + fixed overhead)
        var intercept: Double
        /// time added per layer
        var slope: Double
        /// coefficient of determination; 1 = perfect line (also 1 when all times are equal and fit exactly).
        var r2: Double

        func predict(layers: Double) -> Double { intercept + slope * layers }
    }

    /// Ordinary least squares. Returns nil for fewer than 2 points, mismatched counts,
    /// non-finite values, or fewer than 2 distinct layer counts.
    static func fit(layers: [Double], times: [Double]) -> Fit? {
        guard layers.count == times.count, layers.count >= 2 else { return nil }
        guard (layers + times).allSatisfy({ $0.isFinite }) else { return nil }
        let n = Double(layers.count)
        let mx = layers.reduce(0, +) / n
        let my = times.reduce(0, +) / n
        var sxx = 0.0, sxy = 0.0, syy = 0.0
        for (x, y) in zip(layers, times) {
            sxx += (x - mx) * (x - mx)
            sxy += (x - mx) * (y - my)
            syy += (y - my) * (y - my)
        }
        guard sxx > 0 else { return nil }
        let slope = sxy / sxx
        let intercept = my - slope * mx
        var ssRes = 0.0
        for (x, y) in zip(layers, times) {
            let e = y - (intercept + slope * x)
            ssRes += e * e
        }
        let r2 = syy == 0 ? (ssRes == 0 ? 1 : 0) : 1 - ssRes / syy
        return Fit(intercept: intercept, slope: slope, r2: r2)
    }

    static func extrapolate(_ fit: Fit, toLayers target: Int) -> Double {
        fit.predict(layers: Double(target))
    }

    /// True when time never decreases as layers increase (points sorted by layers first).
    static func isMonotonicIncreasing(layers: [Double], times: [Double]) -> Bool {
        guard layers.count == times.count else { return false }
        let ys = zip(layers, times).sorted { $0.0 < $1.0 }.map { $0.1 }
        return zip(ys, ys.dropFirst()).allSatisfy { $0 <= $1 }
    }

    static func median(_ xs: [Double]) -> Double {
        guard !xs.isEmpty else { return .nan }
        let s = xs.sorted()
        let m = s.count / 2
        return s.count % 2 == 1 ? s[m] : (s[m - 1] + s[m]) / 2
    }

    /// Finds `num_hidden_layers` of the (possibly nested) config object whose `model_type` equals
    /// `modelType`. Nil if there is none.
    static func layerCount(configJSON: Data, modelType: String) -> Int? {
        guard let root = try? JSONSerialization.jsonObject(with: configJSON) else { return nil }
        func search(_ node: Any) -> Int? {
            if let d = node as? [String: Any] {
                if d["model_type"] as? String == modelType, let n = d["num_hidden_layers"] as? Int {
                    return n
                }
                for v in d.values { if let r = search(v) { return r } }
            } else if let a = node as? [Any] {
                for v in a { if let r = search(v) { return r } }
            }
            return nil
        }
        return search(root)
    }
}
