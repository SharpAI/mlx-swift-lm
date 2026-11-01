// SharpAI fork overlay tests for alpamayo2_super checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
@testable import MLXVLM
import XCTest

final class SharpAIAlpamayo2SuperSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() throws {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = try SharpAIAlpamayo2SuperSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testDefaultTableLeavesKeysUnchanged() throws {
        // The default table is empty by design: the released index keeps `expert.` and `vlm.` prefixes.
        XCTAssertTrue(SharpAIAlpamayo2SuperSanitizer.defaultPrefixRemaps.isEmpty)

        let weights: [String: MLXArray] = [
            "vlm.model.embed_tokens.weight": vector(2),
            "action_head.weight": vector(3),
        ]

        let result = try SharpAIAlpamayo2SuperSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(result.keys), ["vlm.model.embed_tokens.weight", "action_head.weight"])
    }

    func testRemapsPrefixWithExplicitTable() throws {
        let weights: [String: MLXArray] = [
            "vlm_config.model.layers.0.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = try SharpAIAlpamayo2SuperSanitizer.sanitize(
            weights, prefixRemaps: [(from: "vlm_config.model.", to: "model.")])

        XCTAssertNotNil(result["model.layers.0.weight"])
        XCTAssertNotNil(result["lm_head.weight"])
        XCTAssertNil(result["vlm_config.model.layers.0.weight"])
    }

    func testFirstMatchingPrefixWins() {
        let key = SharpAIAlpamayo2SuperSanitizer.remappedKey(
            "a.b.c",
            prefixRemaps: [(from: "a.", to: "x."), (from: "a.b.", to: "y.")])

        XCTAssertEqual(key, "x.b.c")
    }

    func testDuplicateAfterRemapThrows() {
        let weights: [String: MLXArray] = [
            "a.weight": vector(1),
            "b.weight": vector(2),
        ]

        XCTAssertThrowsError(
            try SharpAIAlpamayo2SuperSanitizer.sanitize(
                weights, prefixRemaps: [(from: "a.", to: "b.")]))
    }
}
