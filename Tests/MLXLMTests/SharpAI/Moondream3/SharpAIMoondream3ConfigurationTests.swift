// SharpAI fork overlay: moondream3 configuration tests. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIMoondream3ConfigurationTests: XCTestCase {

    func testDecodesNestedTextVisionAndTokenizerKeys() throws {
        let config = try SharpAIMoondream3TestFixtures.decode()

        XCTAssertEqual(config.modelType, "moondream3")
        XCTAssertEqual(config.textConfiguration.dim, 16)
        XCTAssertEqual(config.textConfiguration.heads, 2)
        XCTAssertEqual(config.textConfiguration.layers, 2)
        XCTAssertEqual(config.textConfiguration.moeStartLayer, 1)
        XCTAssertEqual(config.textConfiguration.moeExperts, 4)
        XCTAssertEqual(config.textConfiguration.moeExpertsPerToken, 2)
        XCTAssertEqual(config.textConfiguration.moeExpertInnerDim, 8)
        XCTAssertEqual(config.visionConfiguration.encDim, 8)
        XCTAssertEqual(config.visionConfiguration.gridSize, 2)
        XCTAssertEqual(config.answerTokenId, 3)
    }

    func testImagePrefixLengthIsBosPlusPooledGrid() throws {
        let tiny = try SharpAIMoondream3TestFixtures.decode()
        XCTAssertEqual(tiny.imagePrefixLength, 5)

        let release = try SharpAIMoondream3TestFixtures.decode(#"{"model_type": "moondream3"}"#)
        XCTAssertEqual(release.imagePrefixLength, 730)
    }

    func testMissingConfigUsesVLLMDefaults() throws {
        let config = try SharpAIMoondream3TestFixtures.decode(#"{"model_type": "moondream3"}"#)

        XCTAssertEqual(config.textConfiguration.dim, 2048)
        XCTAssertEqual(config.textConfiguration.layers, 24)
        XCTAssertEqual(config.textConfiguration.moeStartLayer, 4)
        XCTAssertEqual(config.textConfiguration.moeExperts, 64)
        XCTAssertEqual(config.visionConfiguration.encLayers, 27)
        XCTAssertEqual(config.visionConfiguration.cropSize, 378)
        XCTAssertEqual(config.visionConfiguration.maxCrops, 12)
        XCTAssertEqual(config.visionConfiguration.overlapMargin, 4)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsWrongModelType() throws {
        let json = SharpAIMoondream3TestFixtures.tinyJSON.replacingOccurrences(
            of: "\"moondream3\"", with: "\"moondream2\"")
        let config = try SharpAIMoondream3TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsGroupedQueryAttention() throws {
        let json = SharpAIMoondream3TestFixtures.tinyJSON.replacingOccurrences(
            of: "\"n_kv_heads\": 2", with: "\"n_kv_heads\": 1")
        let config = try SharpAIMoondream3TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsPoolSideDifferentFromCropGrid() throws {
        let json = SharpAIMoondream3TestFixtures.tinyJSON.replacingOccurrences(
            of: "\"enc_n_layers\": 2", with: "\"enc_n_layers\": 3")
        let config = try SharpAIMoondream3TestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
