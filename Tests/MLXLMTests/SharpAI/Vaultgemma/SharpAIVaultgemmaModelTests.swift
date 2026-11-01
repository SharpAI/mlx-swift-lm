// SharpAI fork overlay tests for the vaultgemma model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class SharpAIVaultgemmaModelTests: XCTestCase {

    private func tinyModel(tied: Bool = true) throws -> SharpAIVaultgemmaModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIVaultgemmaConfiguration.self,
            from: Data(SharpAIVaultgemmaTestFixture.tinyJSON(tied: tied).utf8))
        return SharpAIVaultgemmaModel(config)
    }

    private func parameterShapes(_ model: SharpAIVaultgemmaModel) -> [String: [Int]] {
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

    func testForwardPassWithCachesRunsSlidingAndFullLayers() throws {
        let model = try tinyModel()
        let cache = try model.newCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)
        XCTAssertEqual(cache[0].maxSize, 4)
        XCTAssertNil(cache[1].maxSize)

        // Longer than the window of 4, so the sliding layer uses its window mask.
        let inputs = MLXArray([0, 1, 2, 3, 4, 5] as [Int32]).reshaped(1, 6)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 6, 100])
    }

    func testForwardPassWithQuantizedFullAttentionCache() throws {
        // Quantized KV caches need a head dim divisible by a supported group size (32, 64 or 128).
        let json = SharpAIVaultgemmaTestFixture.tinyJSON()
            .replacingOccurrences(of: "\"head_dim\": 8", with: "\"head_dim\": 32")
            .replacingOccurrences(of: "\"query_pre_attn_scalar\": 8", with: "\"query_pre_attn_scalar\": 32")
        let config = try JSONDecoder.json5().decode(
            SharpAIVaultgemmaConfiguration.self, from: Data(json.utf8))
        let model = SharpAIVaultgemmaModel(config)
        var cache = try model.newCache(parameters: nil)
        let simple = try XCTUnwrap(cache[1] as? KVCacheSimple)
        cache[1] = try simple.toQuantized(groupSize: 32, bits: 8)

        let inputs = MLXArray([0, 1, 2, 3, 4, 5] as [Int32]).reshaped(1, 6)
        let logits = model(inputs, cache: cache)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 6, 100])
    }

    func testAttentionShapesFollowHeadCounts() throws {
        let shapes = parameterShapes(try tinyModel())

        // 4 query heads and 2 KV heads, head_dim 8, hidden 32.
        XCTAssertEqual(shapes["model.layers.0.self_attn.q_proj.weight"], [32, 32])
        XCTAssertEqual(shapes["model.layers.0.self_attn.k_proj.weight"], [16, 32])
    }

    func testBlocksHaveNoPostNorms() throws {
        let keys = Set(parameterShapes(try tinyModel()).keys)

        XCTAssertTrue(keys.contains("model.layers.0.pre_feedforward_layernorm.weight"))
        XCTAssertFalse(keys.contains { $0.contains("post_attention_layernorm") })
        XCTAssertFalse(keys.contains { $0.contains("post_feedforward_layernorm") })
    }

    func testTiedModelHasNoLMHeadParameter() throws {
        let keys = Set(parameterShapes(try tinyModel(tied: true)).keys)

        XCTAssertFalse(keys.contains { $0.contains("lm_head") })
    }

    func testUntiedModelHasLMHeadParameter() throws {
        let shapes = parameterShapes(try tinyModel(tied: false))

        XCTAssertEqual(shapes["lm_head.weight"], [100, 32])
    }
}
