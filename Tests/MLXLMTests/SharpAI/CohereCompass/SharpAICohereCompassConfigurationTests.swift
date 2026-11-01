// SharpAI fork overlay tests for cohere_compass config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICohereCompassConfigurationTests: XCTestCase {

    private let specJSON = """
        {
            "model_type": "cohere_compass",
            "architectures": ["CohereCompassForConditionalGeneration"]
        }
        """

    func testDecodesModelTypeAndArchitectures() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICohereCompassConfiguration.self, from: Data(specJSON.utf8))

        XCTAssertEqual(config.modelType, "cohere_compass")
        XCTAssertEqual(
            config.architectures, ["CohereCompassForConditionalGeneration"])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingModelTypeUsesDefault() throws {
        // UNVERIFIED: default model_type is an assumption, pinned here so a change is visible.
        let config = try JSONDecoder.json5().decode(
            SharpAICohereCompassConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "cohere_compass")
        XCTAssertNil(config.architectures)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsWrongArchitecture() throws {
        let json = """
            {
                "model_type": "cohere_compass",
                "architectures": ["SomeOtherForConditionalGeneration"]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohereCompassConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testIgnoresUnmodeledSubConfigs() throws {
        // UNVERIFIED: text_config and vision_config layouts are not modeled yet.
        let json = """
            {
                "model_type": "cohere_compass",
                "architectures": ["CohereCompassForConditionalGeneration"],
                "text_config": { "hidden_size": 64 },
                "vision_config": { "depth": 2 }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICohereCompassConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "cohere_compass")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
