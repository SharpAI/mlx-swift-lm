// SharpAI fork overlay tests for hrm_text checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIHrmTextSanitizeTests: XCTestCase {

    private func configuration(tied: Bool) throws -> SharpAIHrmTextConfiguration {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "H_cycles": 2,
                "L_cycles": 3,
                "L_bp_cycles": [0],
                "tie_word_embeddings": \(tied)
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(json.utf8))
    }

    private func weights(_ keys: [String]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: keys.map { ($0, MLXArray([Float(1)])) })
    }

    func testDropsLmHeadWhenEmbeddingsAreTied() throws {
        let config = try configuration(tied: true)
        let input = weights([
            "model.embed_tokens.weight",
            "lm_head.weight",
        ])

        let out = SharpAIHrmTextSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["model.embed_tokens.weight"])
        XCTAssertNil(out["lm_head.weight"])
    }

    func testKeepsLmHeadWhenEmbeddingsAreNotTied() throws {
        let config = try configuration(tied: false)
        let input = weights(["lm_head.weight"])

        let out = SharpAIHrmTextSanitizer.sanitize(input, configuration: config)

        XCTAssertNotNil(out["lm_head.weight"])
    }

    func testOtherKeysKeepHFNames() throws {
        // UNVERIFIED: weight key names are not checked against a real hrm_text checkpoint.
        let config = try configuration(tied: false)
        let keys = [
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.norm.weight",
        ]

        let out = SharpAIHrmTextSanitizer.sanitize(weights(keys), configuration: config)

        XCTAssertEqual(Set(out.keys), Set(keys))
    }
}
