// SharpAI fork overlay tests for molmo2 configuration. Not part of upstream.
// UNVERIFIED: the nested text_config layout is assumed, not read from a real molmo2 config.json.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIMolmo2ConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "molmo2",
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 1024
            },
            "vit_config": {}
        }
        """

    private func decode(_ json: String) throws -> SharpAIMolmo2Configuration {
        try JSONDecoder.json5().decode(
            SharpAIMolmo2Configuration.self, from: Data(json.utf8))
    }

    func testDecodesNestedTextConfig() throws {
        let configuration = try decode(nestedJSON)

        XCTAssertEqual(configuration.modelType, "molmo2")
        XCTAssertTrue(configuration.hasVisionConfig)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testVisionConfigIsOptional() throws {
        let json = nestedJSON.replacingOccurrences(
            of: "\"vit_config\": {}", with: "\"unused\": {}")
        let configuration = try decode(json)

        XCTAssertFalse(configuration.hasVisionConfig)
    }

    func testDefaultsModelTypeWhenMissing() throws {
        let json = nestedJSON.replacingOccurrences(
            of: "\"model_type\": \"molmo2\",", with: "")
        let configuration = try decode(json)

        XCTAssertEqual(configuration.modelType, "molmo2")
    }

    func testRejectsHeadCountNotDivisibleByKVHeads() throws {
        let json = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testRejectsZeroVocabulary() throws {
        let json = nestedJSON.replacingOccurrences(
            of: "\"vocab_size\": 1024", with: "\"vocab_size\": 0")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testMissingTextConfigFailsDecoding() {
        let json = """
            {
                "model_type": "molmo2",
                "vit_config": {}
            }
            """

        XCTAssertThrowsError(try decode(json))
    }

    func testMissingRequiredTextFieldFailsDecoding() {
        let json = nestedJSON.replacingOccurrences(
            of: "\"hidden_size\": 256,", with: "")

        XCTAssertThrowsError(try decode(json))
    }
}
