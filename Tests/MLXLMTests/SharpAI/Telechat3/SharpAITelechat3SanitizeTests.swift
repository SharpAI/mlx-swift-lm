// SharpAI fork overlay tests for telechat3 weight sanitization. Not part of upstream.
// UNVERIFIED: no telechat3 checkpoint key names are available. Tests use Llama-style keys.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAITelechat3SanitizeTests: XCTestCase {

    private func makeModel(tieWordEmbeddings: Bool) throws -> SharpAITelechat3Model {
        let json = """
            {
                "model_type": "telechat3",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 32,
                "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            SharpAITelechat3Configuration.self, from: Data(json.utf8))
        return SharpAITelechat3Model(configuration.llamaConfiguration)
    }

    func testDropsRotaryInvFreq() {
        let sanitized = SharpAITelechat3Sanitizer.sanitize(
            [
                "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
                "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            ],
            tieWordEmbeddings: false)

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() {
        let sanitized = SharpAITelechat3Sanitizer.sanitize(
            [
                "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
                "lm_head.weight": MLXArray.zeros([32, 8]),
            ],
            tieWordEmbeddings: true)

        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsAreNotTied() {
        let sanitized = SharpAITelechat3Sanitizer.sanitize(
            [
                "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
                "lm_head.weight": MLXArray.zeros([32, 8]),
            ],
            tieWordEmbeddings: false)

        XCTAssertNotNil(sanitized["lm_head.weight"])
    }

    func testPrefixMappingRewritesKeys() {
        let sanitized = SharpAITelechat3Sanitizer.sanitize(
            ["transformer.layers.0.mlp.down_proj.weight": MLXArray.zeros([8, 16])],
            tieWordEmbeddings: false,
            mapping: [(from: "transformer.", to: "model.")])

        XCTAssertNil(sanitized["transformer.layers.0.mlp.down_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.mlp.down_proj.weight"])
    }

    func testEmptyDefaultMappingKeepsKeys() {
        XCTAssertTrue(SharpAITelechat3Sanitizer.keyMapping.isEmpty)

        let sanitized = SharpAITelechat3Sanitizer.sanitize(
            ["model.layers.0.mlp.down_proj.weight": MLXArray.zeros([8, 16])],
            tieWordEmbeddings: false)

        XCTAssertNotNil(sanitized["model.layers.0.mlp.down_proj.weight"])
    }

    func testModelSanitizeUsesTelechat3Sanitizer() throws {
        let model = try makeModel(tieWordEmbeddings: true)
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }
}
