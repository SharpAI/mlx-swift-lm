// SharpAI fork overlay tests for the cosmos3_omni configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICosmos3OmniConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "architectures": ["Cosmos3ForConditionalGeneration"],
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "model_type": "cosmos3_omni",
            "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 5120,
                "intermediate_size": 25600,
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 262144,
                "rope_theta": 5000000,
                "rms_norm_eps": 1e-06,
                "rope_scaling": {
                    "mrope_interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "vocab_size": 151936
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 27,
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "out_hidden_size": 5120,
                "num_heads": 16,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 2304,
                "in_channels": 3,
                "deepstack_visual_indexes": [8, 16, 24]
            }
        }
        """

    func testDecodesModelTypeAndArchitectures() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3OmniConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "cosmos3_omni")
        XCTAssertEqual(config.architectures, ["Cosmos3ForConditionalGeneration"])
    }

    func testMissingModelTypeDefaultsToCosmos3Omni() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3OmniConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "cosmos3_omni")
        XCTAssertEqual(config.architectures, [])
    }

    func testValidationAcceptsCosmos3OmniModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3OmniConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3OmniConfiguration.self, from: Data(#"{ "model_type": "llama" }"#.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testQwen3VLAdapterDecodesUnderstandingPath() throws {
        let config = try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(
            from: Data(validJSON.utf8))

        XCTAssertEqual(config.textConfiguration.hiddenSize, 5120)
        XCTAssertEqual(config.textConfiguration.numHiddenLayers, 64)
        XCTAssertEqual(config.textConfiguration.numAttentionHeads, 64)
        XCTAssertEqual(config.textConfiguration.numKeyValueHeads, 8)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeSection, [24, 20, 20])
        XCTAssertEqual(config.visionConfiguration.depth, 27)
        XCTAssertEqual(config.visionConfiguration.outHiddenSize, 5120)
        XCTAssertEqual(config.visionConfiguration.deepstackVisualIndexes, [8, 16, 24])
        XCTAssertEqual(config.imageTokenId, 151_655)
        XCTAssertEqual(config.videoTokenId, 151_656)
    }

    /// Shape of the tiny reference config (rope_parameters and top-level tie_word_embeddings).
    private let tinyRopeJSON = """
        {
          "architectures": ["Cosmos3OmniForConditionalGeneration"],
          "model_type": "cosmos3_omni",
          "tie_word_embeddings": false,
          "text_config": {
            "model_type": "qwen3_vl_text",
            "hidden_size": 32,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 512,
            "rope_parameters": {
              "mrope_interleaved": true,
              "mrope_section": [16, 8, 8],
              "rope_theta": 10000,
              "rope_type": "default"
            },
            "vocab_size": 151669
          },
          "vision_config": {
            "model_type": "qwen3_vl_vision",
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 32,
            "out_hidden_size": 32,
            "num_heads": 4,
            "patch_size": 16,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16
          }
        }
        """

    func testRopeParametersAndTopLevelTieAreReadByQwen3VL() throws {
        let config = try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(
            from: Data(tinyRopeJSON.utf8))

        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeSection, [16, 8, 8])
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeInterleaved, true)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.textConfiguration.vocabSize, 151_669)
    }

    func testJSON5ConfigIsNormalisedLikeStrictJSON() throws {
        let json5 = """
            {
              // comment
              "model_type": "cosmos3_omni",
              "tie_word_embeddings": false,
              "text_config": {
                "model_type": "qwen3_vl_text",
                "hidden_size": 32,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "max_position_embeddings": 512,
                "rope_parameters": {
                  "mrope_interleaved": true,
                  "mrope_section": [16, 8, 8],
                  "rope_theta": 10000,
                },
                "vocab_size": 151669,
              },
              "vision_config": {
                "model_type": "qwen3_vl_vision",
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "num_heads": 4,
                "patch_size": 16,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
              },
            }
            """
        let config = try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(
            from: Data(json5.utf8))

        XCTAssertEqual(config.textConfiguration.ropeTheta, 10_000)
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeSection, [16, 8, 8])
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
    }

    func testQwen3VLAdapterRejectsMissingTextConfig() {
        XCTAssertThrowsError(
            try SharpAICosmos3OmniConfiguration.qwen3VLConfiguration(
                from: Data(#"{ "model_type": "cosmos3_omni" }"#.utf8)))
    }
}
