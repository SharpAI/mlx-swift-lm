// SharpAI fork overlay tests for alpamayo2_super config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIAlpamayo2SuperConfigurationTests: XCTestCase {

    private let flatJSON = """
        {
            "model_type": "alpamayo2_super",
            "vocab_size": 512,
            "trajectory_token_start": 400,
            "trajectory_token_count": 100
        }
        """

    private let wrappedJSON = """
        {
            "model_type": "alpamayo2_super",
            "vlm_config": {
                "vocab_size": 1024,
                "trajectory_token_start": 900,
                "trajectory_token_count": 100
            }
        }
        """

    func testFlatConfigDecodes() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.modelType, "alpamayo2_super")
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertFalse(config.isWrapped)
        XCTAssertEqual(config.trajectoryTokenRange, 400 ..< 500)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testWrappedConfigUnwrapsVLMKeys() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(wrappedJSON.utf8))

        XCTAssertTrue(config.isWrapped)
        XCTAssertEqual(config.vocabularySize, 1024)
        XCTAssertEqual(config.trajectoryTokenRange, 900 ..< 1000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testInnerKeysOverrideTopLevelAndTopLevelIsFallback() throws {
        // UNVERIFIED: precedence of wrapper keys over top-level keys is an assumption.
        let json = """
            {
                "vocab_size": 512,
                "trajectory_token_start": 10,
                "vlm_config": { "vocab_size": 1024 }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.vocabularySize, 1024)
        XCTAssertEqual(config.trajectoryTokenStart, 10)
        XCTAssertNil(config.trajectoryTokenCount)
        XCTAssertNil(config.trajectoryTokenRange)
    }

    func testNullWrapperIsTreatedAsFlat() throws {
        let json = """
            { "vocab_size": 512, "vlm_config": null }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(json.utf8))

        XCTAssertFalse(config.isWrapped)
        XCTAssertEqual(config.vocabularySize, 512)
    }

    func testDefaultModelTypeWhenMissing() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "alpamayo2_super")
        XCTAssertNil(config.vocabularySize)
        XCTAssertNil(config.trajectoryTokenRange)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsRangeOutsideVocab() throws {
        let json = """
            {
                "vocab_size": 512,
                "trajectory_token_start": 500,
                "trajectory_token_count": 100
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsZeroCount() throws {
        let json = """
            { "trajectory_token_start": 4, "trajectory_token_count": 0 }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data(json.utf8))

        XCTAssertNil(config.trajectoryTokenRange)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonPositiveVocab() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIAlpamayo2SuperConfiguration.self, from: Data("{\"vocab_size\": 0}".utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
