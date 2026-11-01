// SharpAI fork overlay tests for granite4_vision configuration. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIGranite4VisionConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "granite4_vision",
            "text_config": {
                "model_type": "granite",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1024,
                "logits_scaling": 1.0,
                "attention_multiplier": 1.0,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "max_position_embeddings": 4096,
                "attention_bias": false,
                "mlp_bias": false,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true
            },
            "vision_config": {}
        }
        """

    private func decode(_ json: String) throws -> SharpAIGranite4VisionConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIGranite4VisionConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesNestedTextConfig() throws {
        let configuration = try decode(nestedJSON)

        XCTAssertEqual(configuration.modelType, "granite4_vision")
        XCTAssertTrue(configuration.hasVisionConfig)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testDecodesFlatTextFieldsWhenTextConfigAbsent() throws {
        let flatJSON = """
            {
                "model_type": "granite4_vision",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1024,
                "logits_scaling": 1.0,
                "attention_multiplier": 1.0,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "max_position_embeddings": 4096,
                "attention_bias": false,
                "mlp_bias": false,
                "tie_word_embeddings": true
            }
            """
        let configuration = try decode(flatJSON)

        XCTAssertFalse(configuration.hasVisionConfig)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testDefaultsModelTypeWhenMissing() throws {
        let json = nestedJSON.replacingOccurrences(
            of: "\"model_type\": \"granite4_vision\",", with: "")
        let configuration = try decode(json)

        XCTAssertEqual(configuration.modelType, "granite4_vision")
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

    func testMissingRequiredTextFieldFailsDecoding() {
        let json = nestedJSON.replacingOccurrences(
            of: "\"hidden_size\": 256,", with: "")

        XCTAssertThrowsError(try decode(json))
    }
}
