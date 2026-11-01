// SharpAI fork overlay tests for the minicpmv4_6 configuration. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv46ConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
            "model_type": "minicpmv4_6",
            "architectures": ["MiniCPMV4_6ForConditionalGeneration"],
            "insert_layer_id": 1,
            "drop_vision_last_layer": false,
            "image_token_id": 248056,
            "video_token_id": 248057,
            "tie_word_embeddings": true,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 128,
                "tie_word_embeddings": true,
                "full_attention_interval": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
            },
            "vision_config": {
                "model_type": "minicpmv4_6_vision",
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 56,
                "patch_size": 14,
                "layer_norm_eps": 1e-6
            }
        }
        """

    func testDecodesTopLevelFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "minicpmv4_6")
        XCTAssertEqual(config.insertLayerId, 1)
        XCTAssertFalse(config.dropVisionLastLayer)
        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertEqual(config.videoTokenId, 248_057)
    }

    func testDecodesVisionAndTextBlocks() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(tinyJSON.utf8))

        XCTAssertEqual(config.visionConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.visionConfiguration.numHiddenLayers, 2)
        XCTAssertEqual(config.visionConfiguration.imageSize, 56)
        XCTAssertEqual(config.visionConfiguration.patchSize, 14)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 4)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 128)
    }

    func testDefaultsWhenOptionalKeysAreMissing() throws {
        let minimal = """
            {
                "text_config": {"model_type": "qwen3_5_text", "hidden_size": 32},
                "vision_config": {}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(minimal.utf8))

        XCTAssertEqual(config.modelType, "minicpmv4_6")
        XCTAssertEqual(config.insertLayerId, -1)
        XCTAssertEqual(config.imageTokenId, 248_056)
        XCTAssertEqual(config.videoTokenId, 248_057)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.imageSize, 980)
    }

    func testValidationRejectsImageSizeNotDivisibleByPatch() throws {
        let bad = tinyJSON.replacingOccurrences(
            of: "\"image_size\": 56", with: "\"image_size\": 57")
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(bad.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testQwen35ConfigurationBuildsFromTextBlock() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIMinicpmv46Configuration.self, from: Data(tinyJSON.utf8))
        let qwen = try config.makeQwen35Configuration()

        XCTAssertEqual(qwen.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(qwen.textConfiguration.hiddenLayers, 4)
        XCTAssertEqual(qwen.textConfiguration.vocabularySize, 128)
    }
}
