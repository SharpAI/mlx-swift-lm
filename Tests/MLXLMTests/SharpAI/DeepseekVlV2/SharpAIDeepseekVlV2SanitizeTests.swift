// SharpAI fork overlay tests for deepseek_vl_v2 key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIDeepseekVlV2SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLanguagePrefixAndKeepsTextKeys() throws {
        let weights: [String: MLXArray] = [
            "language.model.embed_tokens.weight": scalar(),
            "language.model.layers.0.self_attn.q_proj.weight": scalar(),
            "language.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIDeepseekVlV2Sanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "lm_head.weight",
            ])
    }

    func testDropsVisionAndProjectorWeights() throws {
        let weights: [String: MLXArray] = [
            "vision.model.blocks.0.attn.weight": scalar(),
            "projector.layers.weight": scalar(),
            "model.sam_model.patch_embed.weight": scalar(),
            "model.vision_model.embeddings.weight": scalar(),
            "image_newline": scalar(),
            "view_seperator": scalar(),
            "language.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIDeepseekVlV2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "language.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAIDeepseekVlV2Sanitizer.sanitize(weights))
    }
}
