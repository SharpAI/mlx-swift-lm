// SharpAI fork overlay tests for hyperclovax_vision_v2 key sanitization. Not part of upstream.
// UNVERIFIED: these tests pin the guessed prefixes from SharpAIHyperclovaxVisionV2Sanitize.swift,
// not a real checkpoint index. They check the mapping only, not the checkpoint.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIHyperclovaxVisionV2SanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLanguagePrefixAndKeepsTextKeys() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": scalar(),
            "language_model.model.layers.0.self_attn.q_proj.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIHyperclovaxVisionV2Sanitizer.sanitize(weights)

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
            "vision_model.embeddings.patch_embedding.weight": scalar(),
            "vision_tower.encoder.layers.0.weight": scalar(),
            "mm_projector.readout.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIHyperclovaxVisionV2Sanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAIHyperclovaxVisionV2Sanitizer.sanitize(weights))
    }
}
