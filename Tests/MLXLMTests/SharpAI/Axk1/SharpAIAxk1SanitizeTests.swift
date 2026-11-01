// SharpAI fork overlay tests for axk1 sanitize (key mapping). Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIAxk1SanitizeTests: XCTestCase {

    func testDropsPredictionLayerPastConfiguredDepth() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration(numHiddenLayers: 2))
        let sanitized = model.sanitize(weights: [
            "model.layers.1.self_attn.q_a_proj.weight": MLXArray.zeros([1]),
            "model.layers.2.self_attn.q_a_proj.weight": MLXArray.zeros([1]),
            "model.layers.0.rotary_emb.inv_freq": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["model.layers.1.self_attn.q_a_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.self_attn.q_a_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.rotary_emb.inv_freq"])
    }

    func testStacksPerExpertWeightsIntoSwitchMLPOnMoELayers() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            for proj in ["gate_proj", "down_proj", "up_proj"] {
                weights["model.layers.1.mlp.experts.\(e).\(proj).weight"] = MLXArray.zeros([8, 8])
            }
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 8, 8])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 8, 8])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
    }

    func testFillsMissingExpertBiasOnMoELayerOnly() throws {
        // UNVERIFIED: zero fill is an assumption for checkpoints that omit the bias.
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let sanitized = model.sanitize(weights: [:])

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.gate.e_score_correction_bias"]?.shape, [4])
        XCTAssertNil(sanitized["model.layers.0.mlp.gate.e_score_correction_bias"])
    }

    func testKeepsExistingExpertBias() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let bias = MLXArray.ones([4])
        let sanitized = model.sanitize(weights: [
            "model.layers.1.mlp.gate.e_score_correction_bias": bias
        ])

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.gate.e_score_correction_bias"]?.shape, [4])
    }

    func testKeepsPostMLPLayerNormKeyUnchanged() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let sanitized = model.sanitize(weights: [
            "model.layers.1.post_mlp_layernorm.weight": MLXArray.ones([8])
        ])

        XCTAssertEqual(sanitized["model.layers.1.post_mlp_layernorm.weight"]?.shape, [8])
    }

    func testDequantizesFP8BlockScaleAndDropsScaleKey() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.kv_b_proj.weight": MLXArray.ones([128, 128]),
            "model.layers.0.self_attn.kv_b_proj.weight_scale_inv": MLXArray.ones([1, 1]),
        ])

        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.kv_b_proj.weight"]?.shape, [128, 128])
        XCTAssertNil(sanitized["model.layers.0.self_attn.kv_b_proj.weight_scale_inv"])
    }
}
