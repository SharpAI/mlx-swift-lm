// SharpAI fork overlay tests for deepseek_vl_v2 config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXVLM
import XCTest

final class SharpAIDeepseekVlV2ConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "deepseek_vl_v2",
            "language_config": {
                "model_type": "deepseek_v2",
                "vocab_size": 129280,
                "hidden_size": 1280,
                "intermediate_size": 6848,
                "moe_intermediate_size": 896,
                "num_hidden_layers": 12,
                "num_attention_heads": 10,
                "num_key_value_heads": 10,
                "n_routed_experts": 64,
                "n_shared_experts": 2,
                "num_experts_per_tok": 6,
                "kv_lora_rank": null,
                "q_lora_rank": null
            },
            "vision_config": {
                "model_type": "vision"
            }
        }
        """

    func testDecodesNestedLanguageConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "deepseek_vl_v2")
        XCTAssertEqual(config.textDimensions.vocabSize, 129280)
        XCTAssertEqual(config.textDimensions.hiddenSize, 1280)
        XCTAssertEqual(config.textDimensions.numHiddenLayers, 12)
        XCTAssertEqual(config.textDimensions.numAttentionHeads, 10)
        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 10)
        XCTAssertTrue(config.hasVisionConfig)
    }

    func testNullKvLoraRankMeansNoMLA() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(nestedJSON.utf8))

        // Gap: the repo's DeepseekV2 block only builds the MLA attention path.
        XCTAssertNil(config.textDimensions.kvLoraRank)
        XCTAssertFalse(config.textUsesMLA)
    }

    func testSetKvLoraRankMeansMLA() throws {
        let mlaJSON = nestedJSON.replacingOccurrences(
            of: "\"kv_lora_rank\": null", with: "\"kv_lora_rank\": 512")
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(mlaJSON.utf8))

        XCTAssertEqual(config.textDimensions.kvLoraRank, 512)
        XCTAssertTrue(config.textUsesMLA)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let flatJSON = """
            {
                "model_type": "deepseek_vl_v2",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 2)
        XCTAssertFalse(config.hasVisionConfig)
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(nestedJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 10", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIDeepseekVlV2Configuration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
