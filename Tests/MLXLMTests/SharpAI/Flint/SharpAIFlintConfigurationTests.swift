// SharpAI fork overlay tests for flint config decoding and validation. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIFlintConfigurationTests: XCTestCase {

    private let hybridJSON = """
        {
            "model_type": "flint",
            "vocab_size": 512,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "use_sliding_window": true,
            "sliding_window": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention"],
            "no_rope_layer_indices": [2],
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts_per_tok": 2,
            "num_experts": 4,
            "norm_topk_prob": true,
            "router_score_function": "sigmoid",
            "mlp_only_layers": [0]
        }
        """

    private func decode(_ json: String) throws -> SharpAIFlintConfiguration {
        try JSONDecoder.json5().decode(SharpAIFlintConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesAllFields() throws {
        let config = try decode(hybridJSON)

        XCTAssertEqual(config.modelType, "flint")
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.hiddenSize, 32)
        XCTAssertEqual(config.intermediateSize, 64)
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.layerTypes, ["full_attention", "sliding_attention", "full_attention"])
        XCTAssertEqual(config.noRopeLayerIndices, [2])
        XCTAssertEqual(config.mlpOnlyLayers, [0])
        XCTAssertEqual(config.moeIntermediateSize, 16)
        XCTAssertEqual(config.sharedExpertIntermediateSize, 16)
        XCTAssertEqual(config.numExpertsPerTok, 2)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertTrue(config.normTopkProb)
        XCTAssertEqual(config.routerScoreFunction, "sigmoid")
        XCTAssertEqual(config.effectiveSlidingWindow, 4)
        XCTAssertEqual(config.resolvedRopeTheta, 10_000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDefaultsMatchVLLMReference() throws {
        let config = try decode(#"{"model_type": "flint", "num_hidden_layers": 2}"#)

        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.layerTypes, ["full_attention", "full_attention"])
        XCTAssertEqual(config.slidingWindow, 513)
        XCTAssertEqual(config.resolvedRopeTheta, 10_000)
        XCTAssertEqual(config.numExperts, 256)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        XCTAssertEqual(config.moeIntermediateSize, 512)
        XCTAssertEqual(config.sharedExpertIntermediateSize, 1024)
        XCTAssertEqual(config.routerScoreFunction, "sigmoid")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testSlidingWindowDisabledWhenUseSlidingWindowFalse() throws {
        let config = try decode(
            #"{"num_hidden_layers": 1, "use_sliding_window": false, "sliding_window": 4}"#)

        XCTAssertNil(config.effectiveSlidingWindow)
    }

    func testExplicitNullSlidingWindowIsNil() throws {
        let config = try decode(#"{"num_hidden_layers": 1, "sliding_window": null}"#)

        XCTAssertNil(config.slidingWindow)
        XCTAssertNil(config.effectiveSlidingWindow)
    }

    func testRopeParametersThetaTakesPrecedence() throws {
        let config = try decode(
            #"""
            {"num_hidden_layers": 1, "rope_theta": 10000.0,
             "rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}}
            """#)

        XCTAssertEqual(config.resolvedRopeTheta, 500_000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsLayerTypeCountMismatch() throws {
        let config = try decode(
            #"{"num_hidden_layers": 3, "layer_types": ["full_attention"]}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonDefaultRopeType() throws {
        let config = try decode(
            #"{"num_hidden_layers": 1, "rope_parameters": {"rope_type": "yarn"}}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnknownRouterScoreFunction() throws {
        let config = try decode(
            #"{"num_hidden_layers": 1, "router_score_function": "tanh"}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
