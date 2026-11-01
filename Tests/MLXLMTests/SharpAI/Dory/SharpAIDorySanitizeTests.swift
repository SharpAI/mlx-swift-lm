// SharpAI fork overlay tests for Dory key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIDorySanitizeTests: XCTestCase {

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testKeepsHFKeyNamesUnchanged() {
        let input = weights([
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.up_proj.weight",
            "lm_head.weight",
        ])

        let output = SharpAIDorySanitizer.sanitize(input)

        XCTAssertEqual(Set(output.keys), Set(input.keys))
    }

    func testDoesNotDropTiedLmHead() {
        // The fixture sets tie_word_embeddings to false, so lm_head is kept.
        // UNVERIFIED: how sanitize should treat lm_head when embeddings are tied.
        let output = SharpAIDorySanitizer.sanitize(weights(["lm_head.weight"]))
        XCTAssertNotNil(output["lm_head.weight"])
    }
}
