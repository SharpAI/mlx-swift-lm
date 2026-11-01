// SharpAI fork overlay tests for k3_dspark checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIK3DsparkSanitizeTests: XCTestCase {

    private let enabledJSON = """
        {
          "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
          "num_attention_heads": 2, "q_lora_rank": 8, "kv_lora_rank": 4,
          "qk_nope_head_dim": 4, "qk_rope_head_dim": 2, "v_head_dim": 4,
          "vocab_size": 50, "num_target_layers": 2, "target_hidden_size": 6,
          "target_layer_ids": [0, 1], "markov_rank": 4,
          "enable_confidence_head": true, "confidence_head_with_markov": true
        }
        """

    private func configuration(confidence: Bool) throws -> SharpAIK3DsparkConfiguration {
        let json = enabledJSON.replacingOccurrences(
            of: "\"enable_confidence_head\": true",
            with: "\"enable_confidence_head\": \(confidence)")
        return try JSONDecoder.json5().decode(
            SharpAIK3DsparkConfiguration.self, from: Data(json.utf8))
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testAddsModelPrefixToCheckpointKeys() throws {
        let weights: [String: MLXArray] = [
            "layers.0.self_attn.q_a_proj.weight": vector(0),
            "layers.1.mlp.down_proj.weight": vector(1),
            "context_proj.weight": vector(2),
            "context_norm.weight": vector(3),
            "final_norm.weight": vector(4),
            "markov_head.markov_w1.weight": vector(5),
            "markov_head.markov_w2.weight": vector(6),
        ]

        let result = SharpAIK3DsparkSanitizer.sanitize(
            weights, configuration: try configuration(confidence: false))

        XCTAssertEqual(
            Set(result.keys),
            [
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.1.mlp.down_proj.weight",
                "model.context_proj.weight",
                "model.context_norm.weight",
                "model.final_norm.weight",
                "model.markov_head.markov_w1.weight",
                "model.markov_head.markov_w2.weight",
            ])
    }

    func testDropsTargetOwnedAndRotaryKeys() throws {
        let weights: [String: MLXArray] = [
            "embed_tokens.weight": vector(0),
            "lm_head.weight": vector(1),
            "layers.0.self_attn.rotary_emb.inv_freq": vector(2),
            "layers.0.self_attn.kv_b_proj.weight": vector(3),
        ]

        let result = SharpAIK3DsparkSanitizer.sanitize(
            weights, configuration: try configuration(confidence: false))

        XCTAssertNil(result["model.embed_tokens.weight"])
        XCTAssertNil(result["model.lm_head.weight"])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.kv_b_proj.weight"])
    }

    func testKeepsConfidenceHeadOnlyWhenEnabled() throws {
        let weights: [String: MLXArray] = [
            "confidence_head.proj.weight": vector(0),
            "context_proj.weight": vector(1),
        ]

        let enabled = SharpAIK3DsparkSanitizer.sanitize(
            weights, configuration: try configuration(confidence: true))
        XCTAssertNotNil(enabled["model.confidence_head.proj.weight"])

        let disabled = SharpAIK3DsparkSanitizer.sanitize(
            weights, configuration: try configuration(confidence: false))
        XCTAssertNil(disabled["model.confidence_head.proj.weight"])
        XCTAssertNotNil(disabled["model.context_proj.weight"])
    }

    func testDoesNotDoublePrefixKeysThatAlreadyHaveIt() throws {
        let weights: [String: MLXArray] = [
            "model.context_proj.weight": vector(0)
        ]

        let result = SharpAIK3DsparkSanitizer.sanitize(
            weights, configuration: try configuration(confidence: false))

        XCTAssertEqual(Array(result.keys), ["model.context_proj.weight"])
    }
}
