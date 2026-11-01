// SharpAI fork overlay tests for vaultgemma checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIVaultgemmaSanitizeTests: XCTestCase {

    private func makeConfig(tied: Bool) throws -> SharpAIVaultgemmaConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIVaultgemmaConfiguration.self,
            from: Data(SharpAIVaultgemmaTestFixture.tinyJSON(tied: tied).utf8))
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersPostNormsAndTiedLMHead() throws {
        let config = try makeConfig(tied: true)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.post_attention_layernorm.weight": vector(1),
            "model.layers.0.post_feedforward_layernorm.weight": vector(1),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = SharpAIVaultgemmaSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["model.layers.0.post_attention_layernorm.weight"])
        XCTAssertNil(result["model.layers.0.post_feedforward_layernorm.weight"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try makeConfig(tied: false)
        let weights: [String: MLXArray] = ["lm_head.weight": vector(2)]

        let result = SharpAIVaultgemmaSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() throws {
        let config = try makeConfig(tied: false)
        let names = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "model.norm.weight",
        ]
        let weights = Dictionary(uniqueKeysWithValues: names.map { ($0, vector(1)) })

        let result = SharpAIVaultgemmaSanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(Set(result.keys), Set(names))
    }
}
