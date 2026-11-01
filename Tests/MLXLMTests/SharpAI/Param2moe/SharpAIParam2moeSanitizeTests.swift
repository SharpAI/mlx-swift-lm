// SharpAI fork overlay tests for param2moe checkpoint key sanitization. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIParam2moeSanitizeTests: XCTestCase {

    private func model(_ overrides: [String: Any] = [:]) throws -> SharpAIParam2moeModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIParam2moeConfiguration.self,
            from: Data(try SharpAIParam2moeTestFixture.tinyJSON(overrides).utf8))
        return SharpAIParam2moeModel(config)
    }

    /// Checkpoint-shaped weights for the tiny model. Layer 1 is sparse with 4 experts.
    private func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        weights["model.word_embeddings.weight"] = zeros([100, 32])
        weights["lm_head.weight"] = zeros([100, 32])
        weights["model.layers.0.attention.query_key_value.weight"] = zeros([48, 32])
        weights["model.layers.0.attention.dense.weight"] = zeros([32, 32])
        weights["model.layers.0.attention.query_layernorm.weight"] = zeros([8])
        weights["model.layers.0.attention.rotary_emb.inv_freq"] = zeros([4])
        weights["model.layers.0.mlp.gate_proj.weight"] = zeros([64, 32])
        weights["model.layers.0.mlp.up_proj.weight"] = zeros([64, 32])
        weights["model.layers.0.mlp.down_proj.weight"] = zeros([32, 64])

        let prefix = "model.layers.1.mlp"
        weights["\(prefix).gate.weight"] = zeros([4, 32])
        weights["\(prefix).gate.expert_bias"] = MLXArray([1, 2, 3, 6] as [Float])
        for e in 0 ..< 4 {
            weights["\(prefix).experts.\(e).gate_proj.weight"] = zeros([16, 32])
            weights["\(prefix).experts.\(e).up_proj.weight"] = zeros([16, 32])
            weights["\(prefix).experts.\(e).down_proj.weight"] = zeros([32, 16])
        }
        weights["\(prefix).shared_experts.gate_proj.weight"] = zeros([16, 32])
        weights["\(prefix).shared_experts.up_proj.weight"] = zeros([16, 32])
        weights["\(prefix).shared_experts.down_proj.weight"] = zeros([32, 16])
        return weights
    }

    func testExpertsAreStackedIntoSwitchMLP() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [4, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape, [4, 32, 16])
        XCTAssertFalse(sanitized.keys.contains { $0.contains(".experts.0.") })
    }

    func testRouterBiasIsZeroMeanedAndKeepsItsName() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        // [1, 2, 3, 6] has mean 3, so the stored bias is [-2, -1, 0, 3].
        let bias = sanitized["model.layers.1.mlp.gate.expert_bias"]?.asArray(Float.self)
        XCTAssertEqual(bias, [-2, -1, 0, 3])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.weight"])
    }

    func testDenseSharedAndAttentionKeysKeepHFNames() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertNotNil(sanitized["model.word_embeddings.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.attention.query_key_value.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.attention.dense.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.attention.query_layernorm.weight"])
        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"])
    }

    func testRotaryBuffersAreDropped() throws {
        let sanitized = try model().sanitize(weights: checkpointWeights())

        XCTAssertFalse(sanitized.keys.contains { $0.contains("rotary_emb.inv_freq") })
    }

    func testLMHeadIsDroppedWhenEmbeddingsAreTied() throws {
        let tied = try model().sanitize(weights: checkpointWeights())
        XCTAssertNil(tied["lm_head.weight"])

        let untied = try model(["tie_word_embeddings": false])
            .sanitize(weights: checkpointWeights())
        XCTAssertNotNil(untied["lm_head.weight"])
    }
}
