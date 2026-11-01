// SharpAI fork overlay tests for the domynedge model structure and forward pass.
// Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDomynedgeModelTests: XCTestCase {

    func testParameterKeysFollowCheckpointNames() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())
        let keys = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(keys.contains("model.embed_tokens.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.norm.weight"))
        // UNVERIFIED: two-norm pre-norm key names follow the Llama layout.
        XCTAssertTrue(keys.contains("model.layers.0.input_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.post_attention_layernorm.weight"))
        XCTAssertFalse(keys.contains(where: { $0.contains("feedforward_layernorm") }))
        XCTAssertFalse(keys.contains(where: { $0.hasPrefix("lm_head") }))
    }

    func testUntiedConfigurationAddsLMHead() throws {
        let model = SharpAIDomynedgeModel(
            try SharpAIDomynedgeTestFixtures.tinyConfiguration(tied: false))
        let keys = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testRopeIsSkippedOnNopeLayers() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())

        XCTAssertNil(model.model.layers[0].selfAttn.rope)
        XCTAssertNotNil(model.model.layers[1].selfAttn.rope)
        XCTAssertNotNil(model.model.layers[2].selfAttn.rope)
        XCTAssertNil(model.model.layers[3].selfAttn.rope)
    }

    func testPerLayerHeadDimSetsQueryProjectionWidth() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())

        // Layer 0 has head dim 16 with 4 heads; layer 1 has head dim 8.
        XCTAssertEqual(model.model.layers[0].selfAttn.queryProj.weight.dim(0), 64)
        XCTAssertEqual(model.model.layers[1].selfAttn.queryProj.weight.dim(0), 32)
        XCTAssertEqual(model.model.layers[1].selfAttn.keyProj.weight.dim(0), 16)
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }

    func testNewCacheUsesWindowCacheOnSlidingLayers() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())
        let cache = try model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, 4)
        XCTAssertTrue(cache[0] is KVCacheSimple)
        XCTAssertTrue(cache[1] is RotatingKVCache)
        XCTAssertTrue(cache[2] is RotatingKVCache)
        XCTAssertTrue(cache[3] is KVCacheSimple)
    }

    /// Prompt longer than the 4-token window, then one decode step.
    func testWindowedPrefillThenDecodeStep() throws {
        let model = SharpAIDomynedgeModel(try SharpAIDomynedgeTestFixtures.tinyConfiguration())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 6
        let prompt = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let prefill = model(prompt, cache: cache)
        eval(prefill)
        XCTAssertEqual(prefill.shape, [1, tokenCount, 64])

        // UNVERIFIED: the offset of a rotating cache counts all tokens, not only the window.
        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(layerCache.offset, tokenCount, "layer \(i)")
        }

        let next = MLXArray([Int32(7)]).reshaped(1, 1)
        let step = model(next, cache: cache)
        eval(step)
        XCTAssertEqual(step.shape, [1, 1, 64])
    }
}
