// SharpAI fork overlay tests for hyperclovax checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAIHyperclovaxSanitizeTests: XCTestCase {

    private func makeConfig(tie: Bool) throws -> SharpAIHyperclovaxConfiguration {
        let json: [String: Any] = [
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "tie_word_embeddings": tie,
            "use_post_norm": true,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder.json5().decode(SharpAIHyperclovaxConfiguration.self, from: data)
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndTiedLMHead() throws {
        let config = try makeConfig(tie: true)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = SharpAIHyperclovaxSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = ["lm_head.weight": vector(2)]

        let result = SharpAIHyperclovaxSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() throws {
        let config = try makeConfig(tie: false)
        // UNVERIFIED: pins the guessed post norm names from the model file, not the checkpoint.
        // This only confirms the guess; it does not prove the names match the checkpoint.
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.post_attn_layernorm.weight",
            "model.layers.0.post_mlp_layernorm.weight",
            "model.norm.weight",
        ]
        let weights = Dictionary(uniqueKeysWithValues: names.map { ($0, vector(1)) })

        let result = SharpAIHyperclovaxSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(Set(result.keys), Set(names))
    }
}
