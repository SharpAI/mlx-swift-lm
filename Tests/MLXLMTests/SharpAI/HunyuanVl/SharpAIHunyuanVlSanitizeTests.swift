// SharpAI fork overlay tests for hunyuan_vl sanitize. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIHunyuanVlSanitizeTests: XCTestCase {

    func testDropsRotaryBuffers() {
        // UNVERIFIED: the rotary buffer key name is assumed. Confirm against a real index.
        let sanitized = SharpAIHunyuanVlSanitizer.sanitize([
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(sanitized.count, 1)
    }

    func testKeepsHFKeyNamesUnchanged() {
        let sanitized = SharpAIHunyuanVlSanitizer.sanitize([
            "model.embed_tokens.weight": MLXArray.zeros([1]),
            "model.layers.1.mlp.gate_proj.weight": MLXArray.zeros([1]),
            "lm_head.weight": MLXArray.zeros([1]),
        ])

        XCTAssertEqual(sanitized.count, 3)
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.gate_proj.weight"])
        XCTAssertNotNil(sanitized["lm_head.weight"])
    }
}
