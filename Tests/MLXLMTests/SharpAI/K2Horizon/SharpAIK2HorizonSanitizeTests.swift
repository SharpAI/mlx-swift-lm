// SharpAI fork overlay tests for k2_horizon sanitize (key filtering). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIK2HorizonSanitizeTests: XCTestCase {

    func testDropsRotaryInvFreqKeys() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.1.input_layernorm.weight",
            "model.layers.1.post_attention_layernorm.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.norm.weight",
        ]
        var weights: [String: MLXArray] = [:]
        for key in keys {
            weights[key] = MLXArray.zeros([1])
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), Set(keys))
    }

    func testDropsLMHeadWhenEmbeddingsTied() throws {
        let model = SharpAIK2HorizonModel(try SharpAIK2HorizonTestFixtures.configuration())
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([32, 8])
        ])

        XCTAssertNil(sanitized["lm_head.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsUntied() throws {
        let model = SharpAIK2HorizonModel(
            try SharpAIK2HorizonTestFixtures.configuration(
                extra: ", \"tie_word_embeddings\": false"))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([32, 8])
        ])

        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [32, 8])
    }
}
