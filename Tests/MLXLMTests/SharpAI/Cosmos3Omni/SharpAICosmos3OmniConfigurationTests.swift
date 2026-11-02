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

    // MARK: generation configuration

    /// `transformer/config.json` of nvidia/Cosmos3-Nano (flat layout).
    private let transformerJSON = """
        {
          "action_dim": 64, "action_gen": true, "base_fps": 24, "enable_fps_modulation": true,
          "head_dim": 128, "hidden_size": 4096, "intermediate_size": 12288,
          "joint_attn_implementation": "two_way", "latent_channel": 48, "latent_patch_size": 2,
          "max_action_dim": 64, "model_type": "qwen3_vl_text", "num_attention_heads": 32,
          "num_embodiment_domains": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
          "patch_latent_dim": 192, "qk_norm": false, "qk_norm_for_diffusion": true,
          "qk_norm_for_text": true, "rms_norm_eps": 1e-06,
          "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20],
                           "rope_type": "default"},
          "rope_theta": 5000000, "sound_dim": 64, "sound_gen": true, "sound_latent_fps": 25,
          "timestep_scale": 0.001, "unified_3d_mrope_reset_spatial_ids": true,
          "unified_3d_mrope_temporal_modality_margin": 15000, "use_moe": true,
          "video_temporal_causal": false, "vocab_size": 151936
        }
        """

    func testDecodesNanoTransformerConfig() throws {
        let c = try SharpAICosmos3OmniConfiguration.generationConfiguration(
            rootConfig: nil, transformerConfig: Data(transformerJSON.utf8))

        XCTAssertEqual(c.hiddenSize, 4096)
        XCTAssertEqual(c.numAttentionHeads, 32)
        XCTAssertEqual(c.numKeyValueHeads, 8)
        XCTAssertEqual(c.headDim, 128)
        XCTAssertEqual(c.intermediateSize, 12288)
        XCTAssertEqual(c.numHiddenLayers, 36)
        XCTAssertEqual(c.vocabSize, 151936)
        XCTAssertTrue(c.useMoe)
        XCTAssertEqual(c.jointAttnImplementation, "two_way")
        XCTAssertTrue(c.qkNormForDiffusion)
        XCTAssertTrue(c.qkNormForText)
        XCTAssertEqual(c.latentChannel, 48)
        XCTAssertEqual(c.latentPatchSize, 2)
        XCTAssertEqual(c.patchLatentDim, 192)
        XCTAssertEqual(c.maxActionDim, 64)
        XCTAssertEqual(c.numEmbodimentDomains, 32)
        XCTAssertEqual(c.soundDim, 64)
        XCTAssertEqual(c.timestepScale, 0.001)
        XCTAssertEqual(c.ropeTheta, 5_000_000)
        XCTAssertEqual(c.mropeSection, [24, 20, 20])
        XCTAssertTrue(c.mropeInterleaved)
        XCTAssertEqual(c.unified3DMropeTemporalModalityMargin, 15000)
        XCTAssertTrue(c.unified3DMropeResetSpatialIds)
        XCTAssertTrue(c.enableFpsModulation)
        XCTAssertEqual(c.baseFps, 24)
    }

    func testRootConfigSuppliesDimensionsAndNestedGenerationFields() throws {
        // Root layout: dimensions in text_config, generation fields in model.config and
        // model.config.diffusion_expert_config; transformer/config.json absent.
        let root = """
            {
              "model_type": "cosmos3_omni",
              "text_config": {
                "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "vocab_size": 100, "rms_norm_eps": 1e-06,
                "rope_parameters": {"mrope_section": [4, 2, 2], "mrope_interleaved": true,
                                    "rope_theta": 10000}
              },
              "model": { "config": {
                "joint_attn_implementation": "two_way", "max_action_dim": 8,
                "num_embodiment_domains": 3, "sound_dim": 4, "sound_gen": true, "action_gen": true,
                "diffusion_expert_config": { "base_fps": 12,
                  "unified_3d_mrope_temporal_modality_margin": 500 }
              } }
            }
            """

        let c = try SharpAICosmos3OmniConfiguration.generationConfiguration(
            rootConfig: Data(root.utf8))

        XCTAssertEqual(c.hiddenSize, 64)
        XCTAssertEqual(c.ropeTheta, 10_000)
        XCTAssertEqual(c.mropeSection, [4, 2, 2])
        XCTAssertEqual(c.maxActionDim, 8)
        XCTAssertEqual(c.numEmbodimentDomains, 3)
        XCTAssertEqual(c.soundDim, 4)
        XCTAssertEqual(c.baseFps, 12)
        XCTAssertEqual(c.unified3DMropeTemporalModalityMargin, 500)
        // Defaults from the Diffusers signature.
        XCTAssertEqual(c.latentChannel, 48)
        XCTAssertEqual(c.patchLatentDim, 192)
        XCTAssertEqual(c.timestepScale, 0.001)
    }

    func testTransformerConfigOverridesRoot() throws {
        let root = #"{"text_config": {"hidden_size": 1}, "model": {"config": {"sound_dim": 1}}}"#
        let c = try SharpAICosmos3OmniConfiguration.generationConfiguration(
            rootConfig: Data(root.utf8), transformerConfig: Data(transformerJSON.utf8))

        XCTAssertEqual(c.hiddenSize, 4096)
        XCTAssertEqual(c.soundDim, 64)
    }

    func testGenerationConfigRequiresDimensions() {
        XCTAssertThrowsError(
            try SharpAICosmos3OmniConfiguration.generationConfiguration(
                rootConfig: nil, transformerConfig: Data(#"{"hidden_size": 8}"#.utf8)))
        XCTAssertThrowsError(
            try SharpAICosmos3OmniConfiguration.generationConfiguration(rootConfig: nil))
    }

    func testGenerationConfigValidationRejectsInconsistentValues() {
        func decode(_ replacing: String, with new: String) throws {
            let json = transformerJSON.replacingOccurrences(of: replacing, with: new)
            _ = try SharpAICosmos3OmniConfiguration.generationConfiguration(
                rootConfig: nil, transformerConfig: Data(json.utf8))
        }
        // GQA: 32 heads over 5 KV heads.
        XCTAssertThrowsError(try decode("\"num_key_value_heads\": 8", with: "\"num_key_value_heads\": 5"))
        // patch_latent_dim must equal latent_channel x patch^2.
        XCTAssertThrowsError(try decode("\"patch_latent_dim\": 192", with: "\"patch_latent_dim\": 100"))
        // mrope_section must sum to head_dim / 2.
        XCTAssertThrowsError(try decode("[24, 20, 20]", with: "[24, 20, 10]"))
        // Only two_way is implemented.
        XCTAssertThrowsError(try decode("\"two_way\"", with: "\"three_way\""))
        // sound_gen needs sound_dim.
        XCTAssertThrowsError(try decode("\"sound_dim\": 64,", with: ""))
    }
}
