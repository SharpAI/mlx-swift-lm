// SharpAI fork config tests for sarvam_mla. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAISarvamMlaConfigurationTests: XCTestCase {

    func testDecodesSarvam105bFixture() throws {
        let data = Data(SharpAISarvamMlaTestFixture.sarvam105bJSON().utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "sarvam_mla")
        XCTAssertEqual(config.vocabularySize, 262144)
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.intermediateSize, 16384)
        XCTAssertEqual(config.moeIntermediateSize, 2048)
        XCTAssertEqual(config.hiddenLayers, 32)
        XCTAssertEqual(config.attentionHeads, 64)
        XCTAssertEqual(config.kvLoraRank, 512)
        XCTAssertEqual(config.qkNopeHeadDim, 128)
        XCTAssertEqual(config.qkRopeHeadDim, 64)
        XCTAssertEqual(config.vHeadDim, 128)
        XCTAssertEqual(config.qHeadDim, 192)
        XCTAssertEqual(config.headDim, 576)
        XCTAssertTrue(config.useQkNorm)
        XCTAssertEqual(config.firstKDenseReplace, 1)
        XCTAssertEqual(config.numExperts, 128)
        XCTAssertEqual(config.numExpertsPerTok, 8)
        XCTAssertEqual(config.numSharedExperts, 1)
        XCTAssertEqual(config.routedScalingFactor, 2.5, accuracy: 1e-6)
        XCTAssertTrue(config.moeRouterEnableExpertBias)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.ropeScaling?["type"], StringOrNumber.string("deepseek_yarn"))
        XCTAssertEqual(config.ropeScaling?["factor"], StringOrNumber.int(40))
    }

    func testFixtureValidates() throws {
        let data = Data(SharpAISarvamMlaTestFixture.sarvam105bJSON().utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: data)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testTinyConfigDecodesAndValidates() throws {
        let data = Data(SharpAISarvamMlaTestFixture.tinyJSON().utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: data)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingRequiredKeyFailsDecode() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAISarvamMlaConfiguration.self, from: Data("{ \"hidden_size\": 32 }".utf8)))
    }

    func testInconsistentHeadDimFailsValidation() throws {
        let data = Data(SharpAISarvamMlaTestFixture.tinyJSON(headDim: 48).utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: data)
        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAISarvamMlaError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testQHeadDimMismatchFailsValidation() throws {
        let data = Data(SharpAISarvamMlaTestFixture.tinyJSON(qHeadDim: 20).utf8)
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: data)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testWrongModelTypeFailsValidation() throws {
        let json = SharpAISarvamMlaTestFixture.tinyJSON()
            .replacingOccurrences(of: "\"sarvam_mla\"", with: "\"sarvam_moe\"")
        let config = try JSONDecoder.json5().decode(
            SharpAISarvamMlaConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
