import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIPhi4SiglipConfigurationTests: XCTestCase {

    func testDecodesFixtureTextAndVisionFields() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode(
            SharpAIPhi4SiglipTestFixtures.fixtureJSON)

        XCTAssertEqual(config.modelType, "phi4-siglip")
        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.hiddenLayers, 40)
        XCTAssertEqual(config.attentionHeads, 40)
        XCTAssertEqual(config.kvHeads, 10)
        XCTAssertEqual(config.vocabularySize, 100352)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertFalse(config.hasRopeScaling)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.ropeDimensions, 128)

        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.patchSize, 16)
        XCTAssertEqual(config.visionConfiguration.positionCount, 256)
        XCTAssertEqual(config.positionGridSide, 16)
        XCTAssertEqual(config.mmHiddenSize, 1152)
        XCTAssertEqual(config.mmProjectorType, "mlp2x_gelu")
        XCTAssertEqual(config.maxNumPatches, 3600)
        XCTAssertEqual(config.minNumPatches, 256)
        XCTAssertNil(config.imageTokenIndex)
        XCTAssertEqual(config.resolvedImageTokenIndex, 100_256)
    }

    func testResolvedImageTokenIndexPrefersConfigKeyAndNeedsVocabulary() throws {
        let tiny = try SharpAIPhi4SiglipTestFixtures.decode()
        XCTAssertEqual(tiny.resolvedImageTokenIndex, 31)

        let noKey = try SharpAIPhi4SiglipTestFixtures.decode(
            SharpAIPhi4SiglipTestFixtures.tinyJSON
                .replacingOccurrences(of: "\"image_token_index\": 31,", with: ""))
        XCTAssertNil(noKey.resolvedImageTokenIndex)
    }

    func testFixtureConfigValidates() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode(
            SharpAIPhi4SiglipTestFixtures.fixtureJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testTinyConfigDecodesImageTokenIndex() throws {
        let config = try SharpAIPhi4SiglipTestFixtures.decode()

        XCTAssertEqual(config.imageTokenIndex, 31)
        XCTAssertEqual(config.visionConfiguration.patchSize, 2)
        XCTAssertEqual(config.positionGridSide, 2)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsWrongModelType() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"model_type\": \"phi4-siglip\"", with: "\"model_type\": \"phi4\"")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsRopeScaling() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"rope_scaling\": null,",
                with: "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 2.0},")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)

        XCTAssertTrue(config.hasRopeScaling)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnknownProjector() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "mlp2x_gelu", with: "linear")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsProjectorWidthMismatch() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"mm_hidden_size\": 8,", with: "\"mm_hidden_size\": 9,")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsNonSquarePositionTable() throws {
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"num_patches\": 4,", with: "\"num_patches\": 6,")
        let config = try SharpAIPhi4SiglipTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
