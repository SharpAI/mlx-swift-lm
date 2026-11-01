// SharpAI fork overlay tests for the Dory configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDoryConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIDoryConfiguration {
        try JSONDecoder.json5().decode(SharpAIDoryConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesFixtureShape() throws {
        let config = try decode(SharpAIDoryTestFixture.json)

        XCTAssertEqual(config.modelType, "dory")
        XCTAssertEqual(config.architectures, ["DoryForCausalLM"])
        XCTAssertEqual(config.vocabularySize, 131072)
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.hiddenLayers, 72)
        XCTAssertEqual(config.intermediateSize, 7168)
        XCTAssertEqual(config.attentionHeads, 16)
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.maxPositionEmbeddings, 262144)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertEqual(config.slidingWindow, 1024)
        XCTAssertEqual(config.ngptAlphaInit, 0.05, accuracy: 1e-6)
        XCTAssertEqual(config.initializerRange, 0.0177, accuracy: 1e-6)
    }

    func testTiedEmbeddingsFailValidation() throws {
        let json = SharpAIDoryTestFixture.json.replacingOccurrences(
            of: "\"tie_word_embeddings\": false", with: "\"tie_word_embeddings\": true")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIDoryError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    func testDecodesLoopSchedule() throws {
        let config = try decode(SharpAIDoryTestFixture.json)

        XCTAssertEqual(config.nInputLayers, 16)
        XCTAssertEqual(config.nRecurrentLayers, 40)
        XCTAssertEqual(config.nOutputLayers, 16)
        XCTAssertEqual(config.nRecurrentLoops, 2)
        XCTAssertEqual(config.layerTypes.count, 72)
        XCTAssertEqual(config.layerTypes[6], "full_attention")
        XCTAssertEqual(config.layerTypes[1], "mlp")
    }

    func testNullOutputGateDecodesToNil() throws {
        let config = try decode(SharpAIDoryTestFixture.json)
        XCTAssertNil(config.attentionOutputGateLogitScaleInit)
    }

    func testDecodesPerLayerTypeRope() throws {
        let config = try decode(SharpAIDoryTestFixture.json)

        let sliding = try XCTUnwrap(config.ropeParameters["sliding_attention"])
        XCTAssertEqual(sliding.ropeType, "default")
        XCTAssertEqual(sliding.ropeTheta, 10_000, accuracy: 1e-3)
        XCTAssertEqual(sliding.partialRotaryFactor, 1.0)

        let full = try XCTUnwrap(config.ropeParameters["full_attention"])
        XCTAssertEqual(full.ropeTheta, 10_000_000, accuracy: 1)
    }

    func testFixtureValidates() throws {
        let config = try decode(SharpAIDoryTestFixture.json)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMismatchedLayerCountFailsValidation() throws {
        let json = SharpAIDoryTestFixture.json.replacingOccurrences(
            of: "\"num_hidden_layers\": 72", with: "\"num_hidden_layers\": 70")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIDoryError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }

    func testUnknownLayerTypeFailsValidation() throws {
        let json = SharpAIDoryTestFixture.json.replacingOccurrences(
            of: "\"layer_types\": [\"sliding_attention\"", with: "\"layer_types\": [\"linear\"")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIDoryError.invalidConfiguration = error else {
                return XCTFail("unexpected error \(error)")
            }
        }
    }
}
