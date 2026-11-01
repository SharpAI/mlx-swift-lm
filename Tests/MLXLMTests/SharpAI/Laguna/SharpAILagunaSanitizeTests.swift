// SharpAI fork overlay tests for laguna checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILagunaSanitizeTests: XCTestCase {

    private func model(tied: Bool = false) throws -> SharpAILagunaModel {
        let config = try JSONDecoder.json5().decode(
            SharpAILagunaConfiguration.self,
            from: Data(SharpAILagunaTestFixture.tinyJSON(tied: tied).utf8))
        return SharpAILagunaModel(config)
    }

    /// Checkpoint-shaped weights for the tiny model. Layer 1 is the sparse layer (4 experts).
    private func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        weights["model.embed_tokens.weight"] = zeros([100, 32])
        weights["lm_head.weight"] = zeros([100, 32])
        weights["model.layers.0.self_attn.g_proj.weight"] = zeros([4, 32])
        weights["model.layers.0.self_attn.rotary_emb.inv_freq"] = zeros([4])
        weights["model.layers.0.mlp.gate_proj.weight"] = zeros([64, 32])

        let prefix = "model.layers.1.mlp"
        weights["\(prefix).gate.weight"] = zeros([4, 32])
        weights["\(prefix).experts.e_score_correction_bias"] = zeros([4])
        for e in 0 ..< 4 {
            weights["\(prefix).experts.\(e).gate_proj.weight"] = zeros([16, 32])
            weights["\(prefix).experts.\(e).up_proj.weight"] = zeros([16, 32])
            weights["\(prefix).experts.\(e).down_proj.weight"] = zeros([32, 16])
        }
        weights["\(prefix).shared_expert.gate_proj.weight"] = zeros([16, 32])
        return weights
    }

    func testExpertsAreStackedIntoSwitchMLP() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertEqual(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 32, 16])
        XCTAssertFalse(sanitized.keys.contains { $0.contains(".experts.0.") })
    }

    func testRouterBiasMovesOntoGate() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertNil(sanitized["model.layers.1.mlp.experts.e_score_correction_bias"])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.gate.e_score_correction_bias"]?.shape, [4])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.weight"])
    }

    func testDenseAndSharedKeysKeepHFNames() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.g_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.shared_expert.gate_proj.weight"])
    }

    func testRotaryBuffersAreDropped() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertFalse(sanitized.keys.contains { $0.contains("rotary_emb.inv_freq") })
    }

    func testLMHeadIsDroppedWhenEmbeddingsAreTied() throws {
        let tied = try model(tied: true).sanitize(weights: checkpointWeights())
        XCTAssertNil(tied["lm_head.weight"])

        let untied = try model(tied: false).sanitize(weights: checkpointWeights())
        XCTAssertNotNil(untied["lm_head.weight"])
    }
}
