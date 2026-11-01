// SharpAI fork overlay tests for the cosmos3_edge configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICosmos3EdgeConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "cosmos3_edge",
            "architectures": ["Cosmos3EdgeForConditionalGeneration"]
        }
        """

    func testDecodesModelTypeAndArchitectures() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "cosmos3_edge")
        XCTAssertEqual(config.architectures, ["Cosmos3EdgeForConditionalGeneration"])
    }

    func testMissingModelTypeDefaultsToCosmos3Edge() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "cosmos3_edge")
        XCTAssertEqual(config.architectures, [])
    }

    func testValidationAcceptsCosmos3EdgeModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = """
            { "model_type": "llama" }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
