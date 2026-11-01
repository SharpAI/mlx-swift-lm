// SharpAI fork overlay tests for llama_bidirec key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAILlamaBidirecSanitizeTests: XCTestCase {

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testKeepsPrefixedKeysUnchanged() {
        XCTAssertEqual(
            SharpAILlamaBidirecSanitizer.mappedKey("model.layers.0.self_attn.q_proj.weight"),
            "model.layers.0.self_attn.q_proj.weight")
        XCTAssertEqual(
            SharpAILlamaBidirecSanitizer.mappedKey("model.embed_tokens.weight"),
            "model.embed_tokens.weight")
    }

    func testAddsModelPrefixToBareBaseKeys() {
        XCTAssertEqual(
            SharpAILlamaBidirecSanitizer.mappedKey("embed_tokens.weight"),
            "model.embed_tokens.weight")
        XCTAssertEqual(
            SharpAILlamaBidirecSanitizer.mappedKey("layers.1.mlp.up_proj.weight"),
            "model.layers.1.mlp.up_proj.weight")
        XCTAssertEqual(SharpAILlamaBidirecSanitizer.mappedKey("norm.weight"), "model.norm.weight")
    }

    func testDropsRotaryBuffersLMHeadAndScoreHead() {
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(1),
            "lm_head.weight": vector(2),
            "score.weight": vector(3),
            "model.score.weight": vector(4),
            "model.norm.weight": vector(5),
        ]

        let result = SharpAILlamaBidirecSanitizer.sanitize(weights)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNil(result["score.weight"])
        XCTAssertNil(result["model.score.weight"])
        XCTAssertNotNil(result["model.norm.weight"])
    }

    func testBareLayoutIsRemappedToModelTree() {
        let weights: [String: MLXArray] = [
            "embed_tokens.weight": vector(1),
            "layers.0.self_attn.q_proj.weight": vector(2),
            "norm.weight": vector(3),
        ]

        let result = SharpAILlamaBidirecSanitizer.sanitize(weights)

        XCTAssertEqual(result.count, 3)
        XCTAssertNotNil(result["model.embed_tokens.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["model.norm.weight"])
        XCTAssertNil(result["embed_tokens.weight"])
    }
}
