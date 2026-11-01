// SharpAI fork overlay tests for the PanguEmbedded model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class SharpAIPanguembeddedModelTests: XCTestCase {

    private func tinyConfiguration(tied: Bool = false, attentionBias: Bool = true)
        throws -> SharpAIPanguembeddedConfiguration
    {
        let json = """
            {
                "model_type": "PanguEmbedded",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-05,
                "vocab_size": 100,
                "rope_theta": 16000000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": \(tied ? "true" : "false"),
                "bias": \(attentionBias ? "true" : "false")
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIPanguembeddedConfiguration.self, from: Data(json.utf8))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration())
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testAttentionProjectionsCarryBiasWhenEnabled() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration(attentionBias: true))
        let keys = Set(model.parameters().flattened().map(\.0))

        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            XCTAssertTrue(keys.contains("model.layers.0.self_attn.\(proj).bias"), proj)
        }
        XCTAssertFalse(keys.contains("model.layers.0.mlp.gate_proj.bias"))
        XCTAssertFalse(keys.contains("model.layers.0.mlp.down_proj.bias"))
        XCTAssertFalse(keys.contains("model.layers.0.mlp.up_proj.bias"))
    }

    func testAttentionProjectionsHaveNoBiasWhenDisabled() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration(attentionBias: false))
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertFalse(keys.contains("model.layers.0.self_attn.q_proj.bias"))
        XCTAssertFalse(keys.contains("model.layers.0.self_attn.o_proj.bias"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
    }

    func testSanitizeDropsRotaryInvFreq() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([8]),
            "model.embed_tokens.weight": MLXArray.zeros([100, 32]),
        ])
        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testSanitizeFallsBackToEmbeddingForMissingLMHead() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration(tied: false))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([100, 32])
        ])
        XCTAssertEqual(sanitized["lm_head.weight"]?.shape, [100, 32])
    }

    func testSanitizeKeepsExistingLMHead() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration(tied: false))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([100, 32]),
            "lm_head.weight": MLXArray.ones([100, 32]),
        ])
        let head = try XCTUnwrap(sanitized["lm_head.weight"])
        XCTAssertEqual(head.sum().item(Float.self), 3200)
    }

    func testSanitizeDoesNotAddLMHeadWhenTied() throws {
        let model = SharpAIPanguembeddedModel(try tinyConfiguration(tied: true))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([100, 32])
        ])
        XCTAssertNil(sanitized["lm_head.weight"])
    }
}
