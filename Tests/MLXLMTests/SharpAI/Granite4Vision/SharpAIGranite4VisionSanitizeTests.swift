// SharpAI fork overlay tests for granite4_vision key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIGranite4VisionSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testMapsModelLanguageModelPrefixToGraniteNames() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.embed_tokens.weight": scalar(),
            "model.language_model.layers.0.self_attn.q_proj.weight": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAIGranite4VisionSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.norm.weight",
            ])
    }

    func testMapsLanguageModelPrefixAndKeepsLMHead() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.norm.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIGranite4VisionSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight", "lm_head.weight"])
    }

    func testDropsVisionAndProjectorWeights() throws {
        let weights: [String: MLXArray] = [
            "model.vision_tower.encoder.layers.0.weight": scalar(),
            "vision_tower.patch_embed.weight": scalar(),
            "model.multi_modal_projector.linear.weight": scalar(),
            "multi_modal_projector.linear.weight": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAIGranite4VisionSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight"])
    }

    func testDropsRotaryInvFreq() throws {
        let weights: [String: MLXArray] = [
            "model.language_model.layers.0.self_attn.rotary_emb.inv_freq": scalar(),
            "model.language_model.norm.weight": scalar(),
        ]
        let sanitized = try SharpAIGranite4VisionSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.norm.weight"])
    }

    func testDuplicateAfterRemapThrows() {
        let weights: [String: MLXArray] = [
            "model.language_model.norm.weight": scalar(),
            "model.norm.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAIGranite4VisionSanitizer.sanitize(weights))
    }
}
