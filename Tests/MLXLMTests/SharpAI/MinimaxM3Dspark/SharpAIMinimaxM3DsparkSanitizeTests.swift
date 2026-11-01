// SharpAI fork overlay tests for minimax_m3_dspark checkpoint key sanitization. Not part of upstream.
// UNVERIFIED: the key layout is unknown, so these tests only pin the pass-through behaviour.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3DsparkSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testKeepsAllKeysUnchanged() {
        let weights: [String: MLXArray] = [
            "layers.0.self_attn.q_proj.weight": vector(0),
            "model.layers.1.mlp.down_proj.weight": vector(1),
            "embed_tokens.weight": vector(2),
            "lm_head.weight": vector(3),
        ]

        let result = SharpAIMinimaxM3DsparkSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    func testKeepsValuesUnchanged() {
        let weights: [String: MLXArray] = [
            "context_proj.weight": vector(5)
        ]

        let result = SharpAIMinimaxM3DsparkSanitizer.sanitize(weights)

        XCTAssertEqual(result["context_proj.weight"]?.asArray(Float.self), [5, 6])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIMinimaxM3DsparkSanitizer.sanitize([:]).isEmpty)
    }
}
