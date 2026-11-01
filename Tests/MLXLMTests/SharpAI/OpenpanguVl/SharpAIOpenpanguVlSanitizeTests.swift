// SharpAI fork overlay tests for openpangu_vl key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIOpenpanguVlSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLanguagePrefixAndKeepsTextKeys() throws {
        let weights: [String: MLXArray] = [
            "language_model.model.embed_tokens.weight": scalar(),
            "language_model.model.layers.0.self_attn.q_proj.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIOpenpanguVlSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "lm_head.weight",
            ])
    }

    func testDropsVisionWeights() throws {
        let weights: [String: MLXArray] = [
            "visual.blocks.0.attn.qkv.weight": scalar(),
            "model.visual.merger.mlp.0.weight": scalar(),
            "vision_tower.encoder.layers.0.weight": scalar(),
            "language_model.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIOpenpanguVlSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testKeepsUnprefixedTextKeys() throws {
        let weights: [String: MLXArray] = [
            "model.layers.1.mlp.up_proj.weight": scalar()
        ]
        let sanitized = try SharpAIOpenpanguVlSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["model.layers.1.mlp.up_proj.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "language_model.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAIOpenpanguVlSanitizer.sanitize(weights))
    }
}
