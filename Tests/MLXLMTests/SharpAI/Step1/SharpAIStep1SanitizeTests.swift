// SharpAI fork overlay tests for step1 sanitize (key cleanup). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIStep1SanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqAndKeepsOtherKeys() {
        let sanitized = SharpAIStep1Sanitize.sanitize(
            weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
                "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testPassesThroughLMHeadUnchanged() {
        // UNVERIFIED: tie handling is left to the model body, which is not written yet.
        let sanitized = SharpAIStep1Sanitize.sanitize(
            weights: ["lm_head.weight": MLXArray.zeros([8, 4])])

        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [8, 4])
    }
}
