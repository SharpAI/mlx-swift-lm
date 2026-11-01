// SharpAI fork overlay tests for exaone4_5 sanitize. Not part of upstream.

import Foundation
import MLX
import XCTest

import MLXVLM

final class SharpAIExaone45SanitizeTests: XCTestCase {

    func testDropsMTPAndRotaryKeys() {
        // UNVERIFIED: key names are assumed. Confirm against a real checkpoint index.
        let sanitized = SharpAIExaone45Sanitizer.sanitize([
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "mtp.layers.0.self_attn.q_proj.weight": MLXArray.zeros([1]),
            "model.mtp.fc.weight": MLXArray.zeros([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["mtp.layers.0.self_attn.q_proj.weight"])
        XCTAssertNil(sanitized["model.mtp.fc.weight"])
        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testKeepsHFKeyNamesUnchanged() {
        let sanitized = SharpAIExaone45Sanitizer.sanitize([
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
