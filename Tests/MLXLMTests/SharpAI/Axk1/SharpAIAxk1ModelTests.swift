// SharpAI fork overlay tests for axk1 model structure and forward pass. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIAxk1ModelTests: XCTestCase {

    func testPostMLPLayerNormExistsOnlyOnMoELayers() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let keys = Set(model.parameters().flattened().map(\.0))

        // Layer 0 is dense (first_k_dense_replace = 1); layer 1 is MoE.
        XCTAssertFalse(keys.contains("model.layers.0.post_mlp_layernorm.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.post_mlp_layernorm.weight"))
    }

    func testForwardPassProducesLogitsShape() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    func testForwardPassUpdatesCacheOncePerLayer() throws {
        let model = SharpAIAxk1Model(try SharpAIAxk1TestFixtures.configuration())
        let cache = try model.newCache(parameters: nil)
        let tokenCount = 5
        let inputs = MLXArray(Array(Int32(1) ... Int32(tokenCount))).reshaped(1, tokenCount)

        let logits = model(inputs, cache: cache)
        eval(logits)

        for (i, layerCache) in cache.enumerated() {
            XCTAssertEqual(
                layerCache.offset, tokenCount,
                "layer \(i) cache advanced by \(layerCache.offset) for \(tokenCount) tokens")
        }
    }
}
