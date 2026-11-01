// SharpAI fork overlay tests for exaone_moe checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIExaoneMoeSanitizeTests: XCTestCase {

    private func makeModel(tied: Bool) throws -> SharpAIExaoneMoeModel {
        let json = """
            {
                "vocab_size": 32,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "tie_word_embeddings": \(tied),
                "mlp_layer_types": ["dense", "sparse"],
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_intermediate_size": 8,
                "num_shared_experts": 0,
                "n_group": 1,
                "topk_group": 1
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIExaoneMoeConfiguration.self, from: Data(json.utf8))
        return SharpAIExaoneMoeModel(config)
    }

    func testStacksPerExpertWeightsIntoSwitchMLP() throws {
        // UNVERIFIED: per-expert key layout follows the GLM4-MoE convention.
        let model = try makeModel(tied: false)
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 2 {
            for proj in ["gate_proj", "up_proj", "down_proj"] {
                weights["model.layers.1.mlp.experts.\(expert).\(proj).weight"] = MLXArray.zeros([8, 16])
            }
        }

        let result = model.sanitize(weights: weights)

        XCTAssertEqual(
            result["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [2, 8, 16])
        XCTAssertEqual(
            result["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [2, 8, 16])
        XCTAssertNil(result["model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(result["model.layers.1.mlp.experts.1.up_proj.weight"])
    }

    func testDropsMTPRotaryAndKeepsUntiedLMHead() throws {
        let model = try makeModel(tied: false)
        let weights: [String: MLXArray] = [
            "mtp.layers.0.fc.weight": MLXArray.zeros([2]),
            "mtp.fc.weight": MLXArray.zeros([2]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([2]),
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([2]),
            "lm_head.weight": MLXArray.zeros([2]),
        ]

        let result = model.sanitize(weights: weights)

        XCTAssertNil(result["mtp.layers.0.fc.weight"])
        XCTAssertNil(result["mtp.fc.weight"])
        XCTAssertNil(result["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(result["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(result["lm_head.weight"])
    }

    func testDropsTiedLMHead() throws {
        let model = try makeModel(tied: true)
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.zeros([2]),
            "model.embed_tokens.weight": MLXArray.zeros([2]),
        ]

        let result = model.sanitize(weights: weights)

        XCTAssertNil(result["lm_head.weight"])
        XCTAssertNotNil(result["model.embed_tokens.weight"])
    }

    func testDropsLayersBeyondHiddenLayers() throws {
        let model = try makeModel(tied: false)
        let weights: [String: MLXArray] = [
            "model.layers.2.mlp.gate_proj.weight": MLXArray.zeros([2]),
            "model.layers.1.mlp.gate.weight": MLXArray.zeros([2]),
        ]

        let result = model.sanitize(weights: weights)

        XCTAssertNil(result["model.layers.2.mlp.gate_proj.weight"])
        XCTAssertNotNil(result["model.layers.1.mlp.gate.weight"])
    }
}
