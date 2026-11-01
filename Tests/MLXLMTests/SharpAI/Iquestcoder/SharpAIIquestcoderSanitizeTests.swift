// SharpAI fork overlay tests for iquestcoder weight sanitization. Not part of upstream.
//
// No key remapping is needed: the checkpoint keys match LlamaModel. These tests cover the
// Llama sanitize steps that the alias inherits.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIIquestcoderSanitizeTests: XCTestCase {

    private func makeModel(tieWordEmbeddings: Bool) throws -> LlamaModel {
        let json = """
            {
                "model_type": "iquestcoder",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-05,
                "vocab_size": 32,
                "tie_word_embeddings": \(tieWordEmbeddings)
            }
            """
        let config = try JSONDecoder.json5().decode(
            LlamaConfiguration.self, from: Data(json.utf8))
        return LlamaModel(config)
    }

    func testDropsRotaryInvFreq() throws {
        let model = try makeModel(tieWordEmbeddings: false)
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
        ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.q_proj.weight"])
    }

    func testDropsLMHeadWhenEmbeddingsAreTied() throws {
        let model = try makeModel(tieWordEmbeddings: true)
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testKeepsLMHeadWhenEmbeddingsAreNotTied() throws {
        let model = try makeModel(tieWordEmbeddings: false)
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertNotNil(sanitized["lm_head.weight"])
    }
}
