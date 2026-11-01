// SharpAI fork overlay tests for NemotronH_Nano_VL_V2 config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXVLM
import XCTest

final class SharpAINemotronhNanoVlV2ConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "NemotronH_Nano_VL_V2",
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "mamba_num_heads": 32,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "n_groups": 8,
                "intermediate_size": 8192,
                "hybrid_override_pattern": "M*M-"
            },
            "vision_config": {
                "model_type": "radio"
            }
        }
        """

    func testDecodesNestedLlmConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronhNanoVlV2Configuration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "NemotronH_Nano_VL_V2")
        XCTAssertEqual(config.textConfiguration.vocabSize, 131072)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.numHiddenLayers, 4)
        XCTAssertEqual(config.textConfiguration.numAttentionHeads, 16)
        XCTAssertEqual(config.textConfiguration.numKeyValueHeads, 4)
        XCTAssertTrue(config.hasVisionConfig)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let flatJSON = """
            {
                "model_type": "NemotronH_Nano_VL_V2",
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "mamba_num_heads": 8,
                "mamba_head_dim": 32,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 512,
                "hybrid_override_pattern": "M*"
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronhNanoVlV2Configuration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.textConfiguration.hiddenSize, 256)
        XCTAssertFalse(config.hasVisionConfig)
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronhNanoVlV2Configuration.self, from: Data(nestedJSON.utf8))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideKeyValueHeads() throws {
        let badJSON = nestedJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 4", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronhNanoVlV2Configuration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
