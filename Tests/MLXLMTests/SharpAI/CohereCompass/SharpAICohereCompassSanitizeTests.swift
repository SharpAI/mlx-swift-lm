// SharpAI fork overlay tests for cohere_compass checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAICohereCompassSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffers() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
        ]

        let result = SharpAICohereCompassSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(result.count, 1)
    }

    func testKeepsHFKeyNamesUnchanged() {
        // UNVERIFIED: no key remap is known yet. This pins pass-through, including vision keys.
        let weights: [String: MLXArray] = [
            "model.embed_tokens.weight": vector(1),
            "model.vision_tower.blocks.0.attn.qkv.weight": vector(2),
            "lm_head.weight": vector(3),
        ]

        let result = SharpAICohereCompassSanitizer.sanitize(weights)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }

    func testEmptyInputStaysEmpty() {
        XCTAssertTrue(SharpAICohereCompassSanitizer.sanitize([:]).isEmpty)
    }
}
