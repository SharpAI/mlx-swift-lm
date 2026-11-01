// SharpAI fork overlay tests for ovis2_6_moe key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIOvis26MoeSanitizeTests: XCTestCase {

    private func scalar() -> MLXArray {
        MLXArray(Float(1.0))
    }

    func testStripsLlmPrefixAndKeepsTextKeys() throws {
        let weights: [String: MLXArray] = [
            "llm.model.embed_tokens.weight": scalar(),
            "llm.model.layers.0.self_attn.q_proj.weight": scalar(),
            "llm.model.layers.0.mlp.experts.0.gate_proj.weight": scalar(),
            "llm.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIOvis26MoeSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "lm_head.weight",
            ])
    }

    func testDropsVisualTokenizerAndEmbeddingTableWeights() throws {
        let weights: [String: MLXArray] = [
            "visual_tokenizer.vit.encoder.layers.0.weight": scalar(),
            "visual_tokenizer.head.weight": scalar(),
            "vte.weight": scalar(),
            "llm.lm_head.weight": scalar(),
        ]
        let sanitized = try SharpAIOvis26MoeSanitizer.sanitize(weights)

        XCTAssertEqual(Set(sanitized.keys), ["lm_head.weight"])
    }

    func testDuplicateAfterStripThrows() {
        let weights: [String: MLXArray] = [
            "llm.lm_head.weight": scalar(),
            "lm_head.weight": scalar(),
        ]

        XCTAssertThrowsError(try SharpAIOvis26MoeSanitizer.sanitize(weights))
    }
}
