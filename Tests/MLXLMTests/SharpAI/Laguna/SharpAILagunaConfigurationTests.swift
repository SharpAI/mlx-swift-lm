// SharpAI fork overlay tests for the laguna config. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILagunaConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAILagunaConfiguration {
        try JSONDecoder.json5().decode(SharpAILagunaConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesXS2TargetFields() throws {
        let config = try decode(SharpAILagunaTestFixture.xs2JSON())

        XCTAssertEqual(config.modelType, "laguna")
        XCTAssertEqual(config.vocabularySize, 100352)
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.hiddenLayers, 40)
        XCTAssertEqual(config.intermediateSize, 8192)
        XCTAssertEqual(config.attentionHeads, 48)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.numExperts, 256)
        XCTAssertEqual(config.numExpertsPerTok, 8)
        XCTAssertEqual(config.moeIntermediateSize, 512)
        XCTAssertEqual(config.sharedExpertIntermediateSize, 512)
        XCTAssertEqual(config.routedScalingFactor, 2.5, accuracy: 1e-6)
        XCTAssertEqual(config.gating, .perHead)
        XCTAssertFalse(config.swaAttentionSinkEnabled)
        XCTAssertEqual(config.slidingWindow, 512)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testPerLayerHeadsAndLayerKindsFollowFixture() throws {
        let config = try decode(SharpAILagunaTestFixture.xs2JSON())

        XCTAssertEqual(config.attentionHeads(layer: 0), 48)
        XCTAssertEqual(config.attentionHeads(layer: 1), 64)
        XCTAssertFalse(config.isSlidingLayer(0))
        XCTAssertTrue(config.isSlidingLayer(1))
        XCTAssertNil(config.windowSize(layer: 0))
        XCTAssertEqual(config.windowSize(layer: 1), 512)
        XCTAssertFalse(config.isSparseLayer(0))
        XCTAssertTrue(config.isSparseLayer(1))
    }

    func testRopeIsResolvedPerLayerType() throws {
        let config = try decode(SharpAILagunaTestFixture.xs2JSON())

        // Full attention: YaRN on half the head dim.
        XCTAssertEqual(config.ropeType(layer: 0), "yarn")
        XCTAssertEqual(config.ropeTheta(layer: 0) ?? 0, 500000, accuracy: 1)
        XCTAssertEqual(config.rotaryDimensions(layer: 0), 64)
        // No mscale is set. YarnRoPE's own scale must equal attention_factor.
        XCTAssertNil(config.ropeScaling(layer: 0)["mscale"])
        let attentionFactor = config.ropeScaling(layer: 0)["attention_factor"]?.asFloat() ?? 0
        let factor = config.ropeScaling(layer: 0)["factor"]?.asFloat() ?? 1
        let yarnScale = 0.1 * log(factor) + 1
        XCTAssertEqual(attentionFactor, 1.4158883, accuracy: 1e-5)
        XCTAssertEqual(yarnScale, attentionFactor, accuracy: 1e-5)

        // Sliding attention: default RoPE over the full head dim.
        XCTAssertEqual(config.ropeType(layer: 1), "default")
        XCTAssertEqual(config.ropeTheta(layer: 1) ?? 0, 10000, accuracy: 1e-3)
        XCTAssertEqual(config.rotaryDimensions(layer: 1), 128)
    }

    func testXS2FixturePassesValidation() throws {
        let config = try decode(SharpAILagunaTestFixture.xs2JSON())
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDFlashDrafterIsRejected() throws {
        let config = try decode(SharpAILagunaTestFixture.dflashJSON)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testFlatRopeParametersApplyToBothLayerTypes() throws {
        let json = """
            {
              "model_type": "laguna",
              "vocab_size": 100,
              "hidden_size": 32,
              "intermediate_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "layer_types": ["full_attention", "sliding_attention"],
              "sliding_window": 4,
              "rope_theta": 5000.0,
              "partial_rotary_factor": 0.5,
              "mlp_layer_types": ["dense", "dense"]
            }
            """
        let config = try decode(json)

        XCTAssertEqual(config.ropeTheta(layer: 0) ?? 0, 5000, accuracy: 1e-3)
        XCTAssertEqual(config.ropeTheta(layer: 1) ?? 0, 5000, accuracy: 1e-3)
        XCTAssertEqual(config.rotaryDimensions(layer: 0), 4)
        XCTAssertEqual(config.ropeType(layer: 0), "default")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testUnknownGatingModeIsRejected() throws {
        let config = try decode(SharpAILagunaTestFixture.tinyJSON(gating: "\"per-row\""))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testPerElementGatingDecodes() throws {
        let config = try decode(SharpAILagunaTestFixture.tinyJSON(gating: "\"per-element\""))
        XCTAssertEqual(config.gating, .perElement)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingMLPLayerTypesIsRejected() throws {
        var config = try decode(SharpAILagunaTestFixture.tinyJSON())
        config.mlpLayerTypes = nil

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testSwaAttentionSinkIsRejected() throws {
        var config = try decode(SharpAILagunaTestFixture.tinyJSON())
        config.swaAttentionSinkEnabled = true

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testTopKAboveExpertCountIsRejected() throws {
        var config = try decode(SharpAILagunaTestFixture.tinyJSON())
        config.numExpertsPerTok = config.numExperts + 1

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAILagunaError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
