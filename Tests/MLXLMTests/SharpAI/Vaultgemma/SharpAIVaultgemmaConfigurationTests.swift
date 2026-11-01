// SharpAI fork overlay tests for the vaultgemma configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIVaultgemmaConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIVaultgemmaConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIVaultgemmaConfiguration.self, from: Data(json.utf8))
    }

    func testEmptyObjectUsesHFDefaults() throws {
        let config = try decode("{}")

        XCTAssertEqual(config.hiddenLayers, 26)
        XCTAssertEqual(config.hiddenSize, 2304)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.queryPreAttnScalar, 256)
        XCTAssertEqual(config.slidingWindow, 4096)
        XCTAssertEqual(config.attnLogitSoftcapping, 50.0)
        XCTAssertEqual(config.finalLogitSoftcapping, 30.0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.ropeTheta, 10_000)
    }

    func testDefaultLayerTypesAlternateSlidingThenFull() throws {
        let config = try decode("{}")

        XCTAssertEqual(config.layerTypes.count, 26)
        XCTAssertEqual(config.layerTypes[0], "sliding_attention")
        XCTAssertEqual(config.layerTypes[1], "full_attention")
        XCTAssertTrue(config.isSlidingLayer(0))
        XCTAssertFalse(config.isSlidingLayer(1))
    }

    func testTinyConfigDecodesAndValidates() throws {
        let config = try decode(SharpAIVaultgemmaTestFixture.tinyJSON())

        XCTAssertEqual(config.vocabularySize, 100)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.slidingWindow, 4)
        XCTAssertEqual(config.layerTypes, ["sliding_attention", "full_attention"])
        XCTAssertEqual(config.queryPreAttnScalar, 8)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testNullSoftcapDisablesCap() throws {
        let config = try decode(
            #"{"attn_logit_softcapping": null, "final_logit_softcapping": null}"#)

        XCTAssertNil(config.attnLogitSoftcapping)
        XCTAssertNil(config.finalLogitSoftcapping)
    }

    func testNullSlidingWindowMeansNoWindow() throws {
        let config = try decode(
            #"{"num_hidden_layers": 2, "sliding_window": null}"#)

        XCTAssertNil(config.slidingWindow)
        XCTAssertTrue(config.isSlidingLayer(0))
        XCTAssertFalse(config.isSlidingLayer(1))
    }

    func testRopeThetaReadFromRopeParametersWhenTopLevelAbsent() throws {
        let config = try decode(
            #"{"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}}"#)

        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsNonDefaultRopeType() throws {
        let config = try decode(#"{"rope_parameters": {"rope_type": "linear"}}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsLayerTypeCountMismatch() throws {
        let config = try decode(
            #"{"num_hidden_layers": 2, "layer_types": ["full_attention"]}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsHeadCountNotDivisibleByKVHeads() throws {
        let config = try decode(
            #"{"num_attention_heads": 5, "num_key_value_heads": 2}"#)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
