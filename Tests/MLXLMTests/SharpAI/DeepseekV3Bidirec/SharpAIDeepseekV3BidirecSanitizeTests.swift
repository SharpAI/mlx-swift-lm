// SharpAI fork overlay tests for deepseek_v3_bidirec key sanitization. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIDeepseekV3BidirecSanitizeTests: XCTestCase {

    private func makeConfig(layers: Int, experts: Int?) throws
        -> SharpAIDeepseekV3BidirecConfiguration
    {
        var json: [String: Any] = [
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": layers,
            "num_attention_heads": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 4,
        ]
        if let experts {
            json["n_routed_experts"] = experts
        }
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder.json5().decode(
            SharpAIDeepseekV3BidirecConfiguration.self, from: data)
    }

    private func vector(_ value: Float) -> MLXArray {
        MLXArray([value, value + 1])
    }

    func testDropsRotaryBuffersAndMultiTokenPredictionLayers() throws {
        let config = try makeConfig(layers: 1, experts: nil)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": vector(0),
            "model.layers.0.self_attn.q_a_proj.weight": vector(1),
            "model.layers.1.self_attn.q_a_proj.weight": vector(2),
        ]

        let result = try SharpAIDeepseekV3BidirecSanitizer.sanitize(
            weights, configuration: config)

        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_a_proj.weight"])
        XCTAssertNil(result["model.layers.1.self_attn.q_a_proj.weight"])
    }

    func testKeepsHFKeyNamesForUnstackedWeights() throws {
        let config = try makeConfig(layers: 1, experts: nil)
        let weights: [String: MLXArray] = [
            "model.layers.0.self_attn.kv_b_proj.weight": vector(1),
            "lm_head.weight": vector(2),
        ]

        let result = try SharpAIDeepseekV3BidirecSanitizer.sanitize(
            weights, configuration: config)

        XCTAssertNotNil(result["model.layers.0.self_attn.kv_b_proj.weight"])
        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        let config = try makeConfig(layers: 1, experts: 3)
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 3 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.0.mlp.experts.\(expert).\(proj).weight"] =
                    vector(Float(expert))
            }
        }

        let result = try SharpAIDeepseekV3BidirecSanitizer.sanitize(
            weights, configuration: config)

        XCTAssertNil(result["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertEqual(
            result["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape, [3, 2])
        XCTAssertEqual(
            result["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape, [3, 2])
    }

    func testMissingExpertWeightThrows() throws {
        let config = try makeConfig(layers: 1, experts: 2)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.experts.0.gate_proj.weight": vector(0)
        ]

        XCTAssertThrowsError(
            try SharpAIDeepseekV3BidirecSanitizer.sanitize(weights, configuration: config)
        ) { error in
            guard case SharpAIDeepseekV3BidirecError.missingExpertWeight(let key) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(key, "model.layers.0.mlp.experts.1.gate_proj.weight")
        }
    }

    func testDequantizesBlockScaleAndDropsScaleKey() throws {
        let config = try makeConfig(layers: 1, experts: nil)
        let weight = MLXArray.ones([128, 128])
        let scale = MLXArray([Float(2)]).reshaped(1, 1)
        let weights: [String: MLXArray] = [
            "model.layers.0.mlp.gate_proj.weight": weight,
            "model.layers.0.mlp.gate_proj.weight_scale_inv": scale,
        ]

        let result = try SharpAIDeepseekV3BidirecSanitizer.sanitize(
            weights, configuration: config)

        XCTAssertNil(result["model.layers.0.mlp.gate_proj.weight_scale_inv"])
        guard let dequantized = result["model.layers.0.mlp.gate_proj.weight"] else {
            XCTFail("dequantized weight missing")
            return
        }
        XCTAssertEqual(dequantized.shape, [128, 128])
        eval(dequantized)
        XCTAssertEqual(dequantized.max().item(Float.self), 2)
        XCTAssertEqual(dequantized.min().item(Float.self), 2)
    }
}
