import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIEagle25VlConfigurationTests: XCTestCase {

    func testDecodesNestedConfigAndDerivedValues() throws {
        let config = try SharpAIEagle25VlTestFixtures.decode()

        XCTAssertEqual(config.modelType, "eagle_2_5_vl")
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 8)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.visionLayerCount, 2)
        XCTAssertTrue(config.appliesVisionPostLayerNorm)
        XCTAssertEqual(config.pixelShuffleFactor, 2)
        XCTAssertEqual(config.imageSeqLength, 1)
        XCTAssertEqual(config.projectorInputSize, 32)
        XCTAssertEqual(config.imageTokenId, 31)
    }

    func testDefaultsApplyWhenOptionalKeysAreMissing() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"downsample_ratio\": 0.5,", with: "")
            .replacingOccurrences(of: "\"select_layer\": -1,", with: "")
            .replacingOccurrences(of: "\"img_context_token_id\": 31,", with: "")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertEqual(config.downsampleRatio, 0.5)
        XCTAssertEqual(config.selectLayer, -1)
        XCTAssertNil(config.imageTokenId)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testSelectLayerTruncatesVisionAndSkipsPostLayerNorm() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"select_layer\": -1,", with: "\"select_layer\": 0,")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertEqual(config.visionLayerCount, 1)
        XCTAssertFalse(config.appliesVisionPostLayerNorm)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingVisionImageSizeFailsDecoding() {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"image_size\": 28,", with: "")

        XCTAssertThrowsError(try SharpAIEagle25VlTestFixtures.decode(json))
    }

    func testWrongModelTypeFailsValidation() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"model_type\": \"eagle_2_5_vl\"", with: "\"model_type\": \"qwen2\"")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testForceImageSizeMismatchFailsValidation() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(
                of: "\"downsample_ratio\": 0.5,",
                with: "\"downsample_ratio\": 0.5, \"force_image_size\": 56,")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testNonIntegerShuffleFactorFailsValidation() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"downsample_ratio\": 0.5,", with: "\"downsample_ratio\": 0.3,")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTextHeadCountMismatchFailsValidation() throws {
        let json = SharpAIEagle25VlTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"num_attention_heads\": 2,", with: "\"num_attention_heads\": 2,")
            .replacingOccurrences(of: "\"num_attention_heads\": 4,", with: "\"num_attention_heads\": 3,")
        let config = try SharpAIEagle25VlTestFixtures.decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testCreatorBuildsModelFromRegistry() throws {
        let data = Data(SharpAIEagle25VlTestFixtures.tinyJSON.utf8)
        let creator = try XCTUnwrap(SharpAIEagle25VlOverlay.creators["eagle_2_5_vl"])

        let model = try creator(data)

        XCTAssertTrue(model is SharpAIEagle25VlModel)
    }
}
