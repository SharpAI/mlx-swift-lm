// SharpAI fork overlay tests for minicpm_ocr configuration. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmOcrConfigurationTests: XCTestCase {

    func testDecodesModelTypeAndIgnoresUnknownKeys() throws {
        let json = """
            {
              "model_type": "minicpm_ocr",
              "text_config": {"model_type": "minicpm"},
              "vision_config": {"hidden_size": 1152}
            }
            """
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinicpmOcrConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(configuration.modelType, "minicpm_ocr")
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = #"{"model_type": "minicpm"}"#
        let configuration = try JSONDecoder.json5().decode(
            SharpAIMinicpmOcrConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testMissingModelTypeFailsToDecode() {
        let json = #"{"hidden_size": 1024}"#

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIMinicpmOcrConfiguration.self, from: Data(json.utf8)))
    }
}
