// SharpAI fork overlay tests for plamo3 key remapping. Not part of upstream.
// UNVERIFIED: the plamo3 key layout is unknown, so the default table is empty and these tests pin
// the pass-through behaviour and the prefix-rewrite helper with a test-only table.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIPlamo3SanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDefaultTableKeepsAllKeys() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.qkv_proj.weight": vector(0),
            "model.layers.1.mlp.gate_up_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = SharpAIPlamo3Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    func testDefaultTableKeepsValues() {
        let weights: [String: MLXArray] = [
            "model.norm.weight": vector(5)
        ]

        let result = SharpAIPlamo3Sanitizer.sanitize(weights)

        XCTAssertEqual(result["model.norm.weight"]?.asArray(Float.self), [5, 6])
    }

    func testEmptyInputGivesEmptyOutput() {
        XCTAssertTrue(SharpAIPlamo3Sanitizer.sanitize([:]).isEmpty)
    }

    func testPrefixMappingRewritesMatchingKeysOnly() {
        let weights: [String: MLXArray] = [
            "old.layers.0.weight": vector(1),
            "other.bias": vector(2),
        ]

        let result = SharpAIPlamo3Sanitizer.sanitize(
            weights, mapping: [(from: "old.", to: "model.")])

        XCTAssertEqual(Set(result.keys), ["model.layers.0.weight", "other.bias"])
        XCTAssertEqual(result["model.layers.0.weight"]?.asArray(Float.self), [1, 2])
    }

    func testFirstMatchingPrefixWins() {
        let weights: [String: MLXArray] = [
            "a.b.weight": vector(3)
        ]

        let result = SharpAIPlamo3Sanitizer.sanitize(
            weights, mapping: [(from: "a.", to: "x."), (from: "a.b.", to: "y.")])

        XCTAssertEqual(Set(result.keys), ["x.b.weight"])
    }
}
