// SharpAI fork overlay tests for the muse_spark configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMuseSparkConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "architectures": ["MuseSparkForConditionalGeneration"],
            "model_type": "muse_spark"
        }
        """

    func testDecodesModelTypeAndArchitectures() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "muse_spark")
        XCTAssertEqual(config.architectures, ["MuseSparkForConditionalGeneration"])
    }

    func testMissingFieldsUseDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "muse_spark")
        XCTAssertEqual(config.architectures, [])
    }

    func testValidationAcceptsMuseSparkModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMuseSparkConfiguration.self, from: Data(#"{ "model_type": "llama" }"#.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
