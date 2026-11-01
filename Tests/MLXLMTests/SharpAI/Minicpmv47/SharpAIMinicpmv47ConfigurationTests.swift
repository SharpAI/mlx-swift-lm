// SharpAI fork overlay tests for minicpmv4_7 config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv47ConfigurationTests: XCTestCase {

    private let json = SharpAIMinicpmv47Fixtures.tinyJSON

    func testDecodesModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "minicpmv4_7")
    }

    func testDecodesTextAndVisionBlocks() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(json.utf8))

        XCTAssertEqual(config.textConfiguration?.hiddenSize, 32)
        XCTAssertEqual(config.textConfiguration?.vocabularySize, 128)
        XCTAssertEqual(config.visionConfiguration?.patchSize, 14)
        XCTAssertEqual(config.insertLayerId, 0)
        XCTAssertEqual(config.downsampleMode, "16x")
        XCTAssertEqual(config.mergerTimes, 1)
    }

    /// Defaults for the special token ids come from the 1B config.json.
    func testSpecialTokenIdDefaultsMatchCheckpoint() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertEqual(config.videoTokenId, 248_057)
        XCTAssertEqual(config.imageStartId, 248_078)
        XCTAssertEqual(config.imageEndId, 248_079)
        XCTAssertEqual(config.sliceStartId, 248_088)
        XCTAssertEqual(config.sliceEndId, 248_089)
        XCTAssertEqual(config.newlineId, 198)
        XCTAssertEqual(config.insertLayerId, 6)
    }

    func testMissingModelTypeDefaultsToMinicpmv47() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "minicpmv4_7")
    }

    func testWrongModelTypeTypeFailsDecoding() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIMinicpmv47Configuration.self,
                from: Data("{ \"model_type\": 3 }".utf8)))
    }

    func testEmptyModelTypeFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self,
            from: Data("{ \"model_type\": \"\" }".utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingTextConfigFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data("{}".utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testInsertLayerOutsideEncoderFailsValidation() throws {
        let edited = json.replacingOccurrences(
            of: "\"insert_layer_id\": 0", with: "\"insert_layer_id\": 2")
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(edited.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv47Configuration.self, from: Data(json.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
