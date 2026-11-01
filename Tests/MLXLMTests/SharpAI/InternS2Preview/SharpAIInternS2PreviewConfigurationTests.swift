// SharpAI fork overlay tests for the intern_s2_preview configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInternS2PreviewConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "intern_s2_preview",
            "architectures": ["InternS2PreviewForConditionalGeneration"],
            "vocab_size": 512,
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "vocab_size": 512,
                "full_attention_interval": 2,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "num_experts": 0,
                "num_experts_per_tok": 0
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 1,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 64,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64
            }
        }
        """

    func testDecodesAndValidatesInternS2PreviewType() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIInternS2PreviewConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(configuration.modelType, "intern_s2_preview")
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testRejectsOtherModelType() throws {
        let json = """
            { "model_type": "qwen3_5_moe" }
            """
        let configuration = try JSONDecoder.json5().decode(
            SharpAIInternS2PreviewConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testMissingModelTypeFailsDecoding() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIInternS2PreviewConfiguration.self, from: Data("{}".utf8)))
    }

    // The same JSON must also decode into the upstream text and vision config.
    func testSharesQwen35TextAndVisionConfiguration() throws {
        let configuration = try JSONDecoder.json5().decode(
            Qwen35Configuration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(configuration.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(configuration.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(configuration.vocabSize, 512)
    }
}
