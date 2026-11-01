// SharpAI fork overlay tests for jina_embeddings_v5 weight sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJinaEmbeddingsV5SanitizeTests: XCTestCase {

    func testDropsAdapterTensorsFromBaseWeights() {
        let sanitized = SharpAIJinaEmbeddingsV5Sanitize.sanitize(
            [
                "model.layers.0.self_attn.q_proj.lora_A.weight": MLXArray.zeros([1, 8]),
                "model.layers.0.self_attn.q_proj.lora_B.weight": MLXArray.zeros([8, 1]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            ],
            tiedWordEmbeddings: true)

        XCTAssertNil(sanitized["model.layers.0.self_attn.q_proj.lora_A.weight"])
        XCTAssertNil(sanitized["model.layers.0.self_attn.q_proj.lora_B.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsQwen3CheckpointNamesUnchanged() {
        let sanitized = SharpAIJinaEmbeddingsV5Sanitize.sanitize(
            [
                "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
                "model.norm.weight": MLXArray.zeros([8]),
            ],
            tiedWordEmbeddings: true)

        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.norm.weight"])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() {
        let sanitized = SharpAIJinaEmbeddingsV5Sanitize.sanitize(
            [
                "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
                "lm_head.weight": MLXArray.zeros([32, 8]),
            ],
            tiedWordEmbeddings: true)

        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsAreNotTied() {
        let sanitized = SharpAIJinaEmbeddingsV5Sanitize.sanitize(
            [
                "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
                "lm_head.weight": MLXArray.zeros([32, 8]),
            ],
            tiedWordEmbeddings: false)

        XCTAssertNotNil(sanitized["lm_head.weight"])
    }
}
