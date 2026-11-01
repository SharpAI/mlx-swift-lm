// SharpAI fork overlay tests for hyperclovax_vision_v2 config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIHyperclovaxVisionV2ConfigurationTests: XCTestCase {

    // Field names match the conformance fixture config. No checkpoint was read.
    private let nestedJSON = """
        {
            "model_type": "hyperclovax_vision_v2",
            "text_config": {
                "model_type": "hyperclovax",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4
            },
            "vision_config": {
                "model_type": "vision"
            }
        }
        """

    func testDecodesNestedTextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxVisionV2Configuration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "hyperclovax_vision_v2")
        XCTAssertEqual(config.textDimensions.modelType, "hyperclovax")
        XCTAssertEqual(config.textDimensions.vocabSize, 1024)
        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.numHiddenLayers, 2)
        XCTAssertEqual(config.textDimensions.numAttentionHeads, 8)
        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 4)
        XCTAssertTrue(config.hasVisionConfig)
    }

    func testMissingKeyValueHeadsFailsDecoding() {
        let noKVJSON = """
            {
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8
                },
                "vision_config": {}
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIHyperclovaxVisionV2Configuration.self, from: Data(noKVJSON.utf8)))
    }

    func testFlatConfigWithoutTextConfigFailsDecoding() {
        let flatJSON = """
            {
                "model_type": "hyperclovax_vision_v2",
                "vocab_size": 512,
                "hidden_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "vision_config": {}
            }
            """

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIHyperclovaxVisionV2Configuration.self, from: Data(flatJSON.utf8)))
    }

    func testMissingVisionConfigPassesValidation() throws {
        let noVisionJSON = """
            {
                "model_type": "hyperclovax_vision_v2",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 8
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxVisionV2Configuration.self, from: Data(noVisionJSON.utf8))

        XCTAssertFalse(config.hasVisionConfig)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxVisionV2Configuration.self, from: Data(nestedJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 4", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxVisionV2Configuration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
