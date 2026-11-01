// SharpAI fork overlay tests for cohere2_moe checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import XCTest

final class SharpAICohere2MoeSanitizeTests: XCTestCase {

    private func makeConfig(numExperts: Int?, tie: Bool) throws -> SharpAICohere2MoeConfiguration {
        var json: [String: Any] = [
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "vocab_size": 32,
            "tie_word_embeddings": tie,
        ]
        if let numExperts {
            json["num_experts"] = numExperts
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder.json5().decode(SharpAICohere2MoeConfiguration.self, from: data)
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndTiedLMHead() throws {
        let config = try makeConfig(numExperts: nil, tie: true)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
    }

    func testKeepsLMHeadWhenUntied() throws {
        let config = try makeConfig(numExperts: nil, tie: false)
        let weights: [String: MLXArray] = ["lm_head.weight": vector(2)]

        let result = try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        // UNVERIFIED: per-expert key layout follows the GLM4-MoE convention.
        let config = try makeConfig(numExperts: 3, tie: false)
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 3 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.0.mlp.experts.\(expert).\(proj).weight"] =
                    vector(Float(expert))
            }
        }

        let result = try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: config)

        XCTAssertNil(result["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            result["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [3, 2])
        XCTAssertEqual(
            result["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [3, 2])
    }

    func testMissingExpertWeightThrows() throws {
        let config = try makeConfig(numExperts: 2, tie: false)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.0.gate_proj.weight": vector(0)
        ]

        XCTAssertThrowsError(
            try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: config))
    }

    func testUnknownExpertLayoutThrows() throws {
        // UNVERIFIED: a fused expert key is an assumed layout, not a confirmed one.
        let config = try makeConfig(numExperts: 2, tie: false)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.gate_up_proj.weight": vector(0)
        ]

        XCTAssertThrowsError(
            try SharpAICohere2MoeSanitizer.sanitize(weights, configuration: config)
        ) { error in
            guard case SharpAICohere2MoeError.unexpectedExpertLayout = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
