// SharpAI fork overlay tests for the granitemoe_swa model structure and forward pass.
// Not part of upstream. Written without compiling; run them before relying on them.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIGranitemoeSwaModelTests: XCTestCase {

    func testParameterKeysFollowCheckpointNames() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.embed_tokens.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.sinks"))
        XCTAssertTrue(keys.contains("model.layers.0.block_sparse_moe.router.layer.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.shared_mlp.input_linear.weight"))
        XCTAssertTrue(keys.contains("model.norm.weight"))
        XCTAssertFalse(keys.contains(where: { $0.hasPrefix("lm_head") }))
    }

    func testUntiedConfigurationAddsLMHead() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration(
                overriding: ["tie_word_embeddings": false]))
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("lm_head.weight"))
    }

    func testSharedMLPIsOmittedWithoutSharedIntermediateSize() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration(
                overriding: ["shared_intermediate_size": NSNull()]))

        XCTAssertNil(model.model.layers[0].sharedMLP)
        XCTAssertNil(model.model.layers[1].sharedMLP)
    }

    func testRopeIsSkippedOnNopeLayer() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())

        XCTAssertNil(model.model.layers[0].selfAttn.rope)
        XCTAssertNotNil(model.model.layers[1].selfAttn.rope)
    }

    func testSinkParameterIsHeadShapedAndUsedOnlyWhenEnabled() throws {
        let withSinks = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())
        XCTAssertTrue(withSinks.model.layers[0].selfAttn.hasSinks)
        XCTAssertEqual(withSinks.model.layers[0].selfAttn.sinks.shape, [4])

        let withoutSinks = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration(
                overriding: ["attention_sinks": false]))
        XCTAssertFalse(withoutSinks.model.layers[0].selfAttn.hasSinks)
    }

    func testNewCacheHasOneEntryPerLayer() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())

        let cache = try model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, 2)
    }

    func testForwardPassProducesLogitsShape() throws {
        // Three tokens with a window of 2 exercise the sliding mask. Random weights only.
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let logits = model(inputs, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }

    func testForwardPassWithCacheProducesLogitsShape() throws {
        let model = SharpAIGranitemoeSwaModel(
            try SharpAIGranitemoeSwaTestFixtures.configuration())
        let inputs = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)
        let cache = try model.newCache(parameters: nil)

        let logits = model(inputs, cache: cache)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 3, 64])
    }
}
