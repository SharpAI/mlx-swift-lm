// SharpAI fork overlay tests for the param2moe model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.
// UNVERIFIED: no test exercises grouped routing numerically. The group mask is only run for shape.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIParam2moeModelTests: XCTestCase {

    private func tinyModel(_ overrides: [String: Any] = [:]) throws -> SharpAIParam2moeModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIParam2moeConfiguration.self,
            from: Data(try SharpAIParam2moeTestFixture.tinyJSON(overrides).utf8))
        return SharpAIParam2moeModel(config)
    }

    private func parameterShapes(_ model: SharpAIParam2moeModel) -> [String: [Int]] {
        Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = try tinyModel()
        let inputs = MLXArray([0, 1, 2, 3, 4] as [Int32]).reshaped(1, 5)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 100])
    }

    func testForwardPassWithTiedEmbeddingsAndCacheProducesLogitsShape() throws {
        let model = try tinyModel(["tie_word_embeddings": true])
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 3)

        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testForwardPassWithUntiedHeadProducesLogitsShape() throws {
        let model = try tinyModel(["tie_word_embeddings": false])
        let inputs = MLXArray([0, 1, 2] as [Int32]).reshaped(1, 3)
        let logits = model(inputs, cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, 100])
    }

    func testDenseLayerUsesMLPAndSparseLayersUseRouter() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertEqual(shapes["model.layers.0.mlp.gate_proj.weight"], [64, 32])
        XCTAssertNil(shapes["model.layers.0.mlp.gate.weight"])
        XCTAssertEqual(shapes["model.layers.1.mlp.gate.weight"], [4, 32])
        XCTAssertEqual(shapes["model.layers.1.mlp.gate.expert_bias"], [4])
        XCTAssertEqual(shapes["model.layers.1.mlp.switch_mlp.gate_proj.weight"], [4, 16, 32])
        XCTAssertEqual(shapes["model.layers.2.mlp.switch_mlp.down_proj.weight"], [4, 32, 16])
        XCTAssertEqual(shapes["model.layers.1.mlp.shared_experts.gate_proj.weight"], [16, 32])
    }

    func testAttentionUsesFusedQKVWithQKNormAndNoBias() throws {
        let shapes = parameterShapes(try tinyModel())

        // (4 query heads + 2 * 2 kv heads) * head_dim 8 = 64 rows.
        XCTAssertEqual(shapes["model.layers.0.attention.query_key_value.weight"], [64, 32])
        XCTAssertEqual(shapes["model.layers.0.attention.dense.weight"], [32, 32])
        XCTAssertEqual(shapes["model.layers.0.attention.query_layernorm.weight"], [8])
        XCTAssertEqual(shapes["model.layers.0.attention.key_layernorm.weight"], [8])
        XCTAssertNil(shapes["model.layers.0.attention.query_key_value.bias"])
    }

    func testEmbeddingUsesWordEmbeddingsName() throws {
        let shapes = parameterShapes(try tinyModel())

        XCTAssertEqual(shapes["model.word_embeddings.weight"], [100, 32])
        XCTAssertNil(shapes["lm_head.weight"])
    }
}
