// SharpAI fork overlay tests for llama_nemotron_vl_rerank config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXVLM
import XCTest

final class SharpAILlamaNemotronVlRerankConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "llama_nemotron_vl_rerank",
            "architectures": ["LlamaNemotronVLForSequenceClassification"],
            "llm_config": {
                "model_type": "llama",
                "vocab_size": 128256,
                "hidden_size": 2048,
                "num_hidden_layers": 16,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "intermediate_size": 8192,
                "rms_norm_eps": 1e-5
            },
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 32,
                "patch_size": 16,
                "image_size": 32
            }
        }
        """

    private let flatJSON = """
        {
            "model_type": "llama_nemotron_vl_rerank",
            "architectures": ["LlamaNemotronVLForSequenceClassification"],
            "vocab_size": 1024,
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 512,
            "rms_norm_eps": 1e-5
        }
        """

    func testDecodesNestedLlmConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "llama_nemotron_vl_rerank")
        XCTAssertEqual(config.architectures, ["LlamaNemotronVLForSequenceClassification"])
        XCTAssertTrue(config.hasVisionConfig)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(flatJSON.utf8))

        XCTAssertFalse(config.hasVisionConfig)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 8", with: "\"num_key_value_heads\": 5")
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingRequiredTextFieldFailsDecoding() {
        let missingJSON = flatJSON.replacingOccurrences(
            of: "\"hidden_size\": 256,", with: "")

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(missingJSON.utf8)))
    }

    func testMalformedVisionConfigFailsDecoding() {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_hidden_layers\": 1,", with: "")

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAILlamaNemotronVlRerankConfiguration.self, from: Data(badJSON.utf8)))
    }
}
