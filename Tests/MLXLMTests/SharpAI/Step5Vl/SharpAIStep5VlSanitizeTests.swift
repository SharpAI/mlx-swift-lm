// SharpAI fork overlay tests for step5_vl sanitize (key cleanup). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIStep5VlSanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqAndKeepsOtherKeys() {
        let sanitized = SharpAIStep5VlSanitize.sanitize(
            weights: [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
                "model.embed_tokens.weight": MLXArray.zeros([8, 4]),
            ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testKeepsVisionTowerAndProjectorKeysUnchanged() {
        // Key names come from the step5_vl fixture's quantization module list.
        let names = [
            "vision_model.transformer.resblocks.0.attn.qkv_proj.weight",
            "vision_model.transformer.resblocks.46.mlp.fc2.weight",
            "vit_large_projector.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for name in names {
            weights[name] = MLXArray.zeros([2, 2])
        }

        let sanitized = SharpAIStep5VlSanitize.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), Set(names))
    }

    func testPassesThroughLMHeadUnchanged() {
        // UNVERIFIED: tie handling is left to the model body, which is not written yet.
        let sanitized = SharpAIStep5VlSanitize.sanitize(
            weights: ["lm_head.weight": MLXArray.zeros([8, 4])])

        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [8, 4])
    }

    func testPassesThroughMtpKeysUnchanged() {
        // UNVERIFIED: the model body is expected to skip these. Sanitize keeps them.
        let sanitized = SharpAIStep5VlSanitize.sanitize(
            weights: ["model.layers.3.mtp_block.mlp.down_proj.weight": MLXArray.zeros([4, 4])])

        XCTAssertNotNil(sanitized["model.layers.3.mtp_block.mlp.down_proj.weight"])
    }
}
