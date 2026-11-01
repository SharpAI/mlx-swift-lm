// SharpAI fork overlay tests for the hy_v3 config. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV3ConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIHyV3Configuration {
        try JSONDecoder.json5().decode(SharpAIHyV3Configuration.self, from: Data(json.utf8))
    }

    func testDecodesHy3FixtureFields() throws {
        let config = try decode(SharpAIHyV3TestFixture.hy3JSON)

        XCTAssertEqual(config.modelType, "hy_v3")
        XCTAssertEqual(config.vocabularySize, 120832)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.intermediateSize, 13312)
        XCTAssertEqual(config.moeIntermediateSize, 1536)
        XCTAssertEqual(config.hiddenLayers, 80)
        XCTAssertEqual(config.attentionHeads, 64)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.ropeTheta, 11158840.0, accuracy: 1)
        XCTAssertEqual(config.ropeType, "default")
        XCTAssertEqual(config.maxPositionEmbeddings, 262144)
        XCTAssertTrue(config.qkNorm)
        XCTAssertFalse(config.attentionBias)
        XCTAssertEqual(config.firstKDenseReplace, 1)
        XCTAssertEqual(config.numExperts, 192)
        XCTAssertEqual(config.numExpertsPerTok, 8)
        XCTAssertEqual(config.numSharedExperts, 1)
        XCTAssertTrue(config.moeRouterUseSigmoid)
        XCTAssertTrue(config.moeRouterEnableExpertBias)
        XCTAssertTrue(config.routeNorm)
        XCTAssertEqual(config.routerScalingFactor, 2.826, accuracy: 1e-5)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
    }

    func testHy3FixturePassesValidation() throws {
        let config = try decode(SharpAIHyV3TestFixture.hy3JSON)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testTopLevelRopeThetaIsAcceptedWhenRopeParametersAbsent() throws {
        let json = """
            {
                "vocab_size": 100,
                "hidden_size": 32,
                "intermediate_size": 64,
                "moe_intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "rope_theta": 5000.0,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "moe_router_enable_expert_bias": true
            }
            """
        let config = try decode(json)
        XCTAssertEqual(config.ropeTheta, 5000.0, accuracy: 1e-3)
        XCTAssertEqual(config.ropeType, "default")
        XCTAssertEqual(config.modelType, "hy_v3")
    }

    func testSoftmaxRouterIsRejected() throws {
        let config = try decode(SharpAIHyV3TestFixture.tinyJSON(sigmoid: false))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIHyV3Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testTopKAboveExpertCountIsRejected() throws {
        var config = try decode(SharpAIHyV3TestFixture.tinyJSON())
        config.numExpertsPerTok = config.numExperts + 1

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIHyV3Error.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
