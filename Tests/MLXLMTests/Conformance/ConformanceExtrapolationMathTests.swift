// Unit tests for the extrapolation maths with hand-made numbers. Pure Swift, no GPU.

import Foundation
import Testing

private typealias X = SharpAIOverlayExtrapolation

struct ConformanceExtrapolationMathTests {

    @Test("exact line y = 3 + 2x is recovered, r2 = 1")
    func exactLine() throws {
        let f = try #require(X.fit(layers: [2, 4, 8], times: [7, 11, 19]))
        #expect(abs(f.intercept - 3) < 1e-12)
        #expect(abs(f.slope - 2) < 1e-12)
        #expect(abs(f.r2 - 1) < 1e-12)
    }

    @Test("extrapolation evaluates the fitted line at the target layer count")
    func extrapolates() throws {
        let f = try #require(X.fit(layers: [2, 4, 8], times: [7, 11, 19]))
        #expect(abs(X.extrapolate(f, toLayers: 48) - 99) < 1e-9)
    }

    @Test("noisy points: hand-computed least squares and r2")
    func noisy() throws {
        // x = 1,2,3  y = 1,3,2  -> slope 0.5, intercept 1, ssRes = 1.5, syy = 2, r2 = 0.25
        let f = try #require(X.fit(layers: [1, 2, 3], times: [1, 3, 2]))
        #expect(abs(f.slope - 0.5) < 1e-12)
        #expect(abs(f.intercept - 1.0) < 1e-12)
        #expect(abs(f.r2 - 0.25) < 1e-12)
    }

    @Test("degenerate inputs give nil")
    func degenerate() {
        #expect(X.fit(layers: [2], times: [1]) == nil)
        #expect(X.fit(layers: [2, 2], times: [1, 2]) == nil)
        #expect(X.fit(layers: [2, 4], times: [1]) == nil)
        #expect(X.fit(layers: [2, 4], times: [1, .nan]) == nil)
    }

    @Test("monotonicity check sorts by layers and rejects a dip")
    func monotonic() {
        #expect(X.isMonotonicIncreasing(layers: [8, 2, 4], times: [19, 7, 11]))
        #expect(!X.isMonotonicIncreasing(layers: [2, 4, 8], times: [7, 12, 11]))
    }

    @Test("median of odd, even and empty")
    func median() {
        #expect(X.median([3, 1, 2]) == 2)
        #expect(X.median([4, 1, 2, 3]) == 2.5)
        #expect(X.median([]).isNaN)
    }

    @Test("layerCount finds the nested config of the requested model_type")
    func layerCount() {
        let json =
            #"{"model_type":"outer","num_hidden_layers":9,"llm_config":{"model_type":"qwen3_moe","num_hidden_layers":48}}"#
        #expect(X.layerCount(configJSON: Data(json.utf8), modelType: "qwen3_moe") == 48)
        #expect(X.layerCount(configJSON: Data(json.utf8), modelType: "outer") == 9)
        #expect(X.layerCount(configJSON: Data(json.utf8), modelType: "nope") == nil)
    }
}
