// SharpAI fork overlay tests for jais2 sanitize (key cleanup). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIJais2SanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqAndKeepsOtherKeys() {
        let sanitized = SharpAIJais2Weights.sanitize(
            weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
                "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            ], tieWordEmbeddings: false)

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsAreNotTied() {
        let sanitized = SharpAIJais2Weights.sanitize(
            weights: ["lm_head.weight": MLXArray.zeros([8, 4])], tieWordEmbeddings: false)

        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [8, 4])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() {
        let sanitized = SharpAIJais2Weights.sanitize(
            weights: [
                "lm_head.weight": MLXArray.zeros([8, 4]),
                "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            ], tieWordEmbeddings: true)

        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }
}
