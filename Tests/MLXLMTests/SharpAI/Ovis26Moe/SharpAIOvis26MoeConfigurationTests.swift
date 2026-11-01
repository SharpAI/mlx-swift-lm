// SharpAI fork overlay tests for ovis2_6_moe config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIOvis26MoeConfigurationTests: XCTestCase {

    // UNVERIFIED: field names are taken from the Ovis2 layout, not from a real checkpoint config.
    private let nestedJSON = """
        {
            "model_type": "ovis2_6_moe",
            "llm_config": {
                "model_type": "qwen3_moe",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_experts": 8,
                "num_experts_per_tok": 2
            },
            "vit_config": {
                "model_type": "siglip2_navit"
            },
            "visual_vocab_size": 65536
        }
        """

    func testDecodesNestedLlmConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "ovis2_6_moe")
        XCTAssertEqual(config.textDimensions.modelType, "qwen3_moe")
        XCTAssertEqual(config.textDimensions.vocabSize, 1024)
        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.numHiddenLayers, 2)
        XCTAssertEqual(config.textDimensions.numAttentionHeads, 8)
        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 4)
        XCTAssertEqual(config.textDimensions.numExperts, 8)
        XCTAssertEqual(config.textDimensions.numExpertsPerTok, 2)
        XCTAssertTrue(config.hasVisionConfig)
        XCTAssertEqual(config.visualVocabSize, 65536)
    }

    func testTextConfigAliasIsAccepted() throws {
        let aliasJSON = nestedJSON.replacingOccurrences(of: "llm_config", with: "text_config")
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(aliasJSON.utf8))

        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
    }

    func testMissingKeyValueHeadsDefaultsToAttentionHeads() throws {
        let noKVJSON = """
            {
                "llm_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_experts": 8,
                    "num_experts_per_tok": 2
                },
                "vit_config": {}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(noKVJSON.utf8))

        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 8)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let flatJSON = """
            {
                "model_type": "ovis2_6_moe",
                "vocab_size": 512,
                "hidden_size": 128,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_experts": 4,
                "num_experts_per_tok": 1,
                "vision_config": {}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.textDimensions.hiddenSize, 128)
        XCTAssertEqual(config.textDimensions.vocabSize, 512)
        XCTAssertTrue(config.hasVisionConfig)
        XCTAssertNil(config.visualVocabSize)
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(nestedJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingVisionConfigFailsValidation() throws {
        let noVisionJSON = """
            {
                "model_type": "ovis2_6_moe",
                "llm_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_experts": 8,
                    "num_experts_per_tok": 2
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(noVisionJSON.utf8))

        XCTAssertFalse(config.hasVisionConfig)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testDenseConfigWithoutExpertsFailsValidation() throws {
        let denseJSON = nestedJSON.replacingOccurrences(
            of: "\"num_experts\": 8", with: "\"num_experts\": 0")
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(denseJSON.utf8))

        XCTAssertEqual(config.textDimensions.numExperts, 0)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTopKAboveExpertCountFailsValidation() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_experts_per_tok\": 2", with: "\"num_experts_per_tok\": 16")
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 4", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIOvis26MoeConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
