// SharpAI fork overlay tests for mistral_large4 checkpoint key sanitization. Not part of upstream.
// UNVERIFIED: key prefixes are assumed, not taken from a Mistral Large 4 checkpoint index.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMistralLarge4SanitizeTests: XCTestCase {

    private func makeConfig(tie: Bool) throws -> SharpAIMistralLarge4Configuration {
        let json = """
            {
              "model_type": "mistral_large4",
              "text_config": {"hidden_size": 8, "tie_word_embeddings": \(tie)}
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIMistralLarge4Configuration.self, from: Data(json.utf8))
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndTiedLMHead() throws {
        let config = try makeConfig(tie: true)
        let weights: [String: MLXArray] = [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "language_model.model.layers.0.self_attn.q_proj.weight": vector(1),
            "language_model.lm_head.weight": vector(2),
        ]

        let result = SharpAIMistralLarge4Sanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["language_model.lm_head.weight"])
        XCTAssertNotNil(result["language_model.model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = ["language_model.lm_head.weight": vector(2)]

        let result = SharpAIMistralLarge4Sanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["language_model.lm_head.weight"])
    }

    func testKeepsHFKeyNamesUnchanged() throws {
        let config = try makeConfig(tie: false)
        let weights: [String: MLXArray] = [
            "vision_tower.patch_conv.weight": vector(3),
            "multi_modal_projector.linear_1.weight": vector(4),
        ]

        let result = SharpAIMistralLarge4Sanitizer.sanitize(weights, configuration: config)

        XCTAssertEqual(Set(result.keys), Set(weights.keys))
    }
}
