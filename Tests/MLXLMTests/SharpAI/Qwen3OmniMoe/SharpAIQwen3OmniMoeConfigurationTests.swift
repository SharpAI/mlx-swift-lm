// SharpAI fork overlay tests for qwen3_omni_moe configuration. Inline configs only, no weights.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

/// Inline configs for qwen3_omni_moe tests. Shapes are tiny. Token ids are 30 to 33.
enum SharpAIQwen3OmniMoeTestConfig {

    /// Thinker config. `flat` puts the blocks at top level instead of under `thinker_config`.
    /// `outputDim` must equal the text hidden size (32) to validate.
    static func tiny(flat: Bool = false, outputDim: Int = 32, tokenIds: Bool = true) -> String {
        let ids =
            tokenIds
            ? """
            "audio_token_id": 30,
            "image_token_id": 31,
            "video_token_id": 32,
            "vision_start_token_id": 33,
            """
            : ""
        let blocks = """
            "audio_config": {
              "model_type": "qwen3_omni_moe_audio_encoder",
              "d_model": 16,
              "encoder_layers": 1,
              "encoder_attention_heads": 2,
              "encoder_ffn_dim": 32,
              "num_mel_bins": 8,
              "max_source_positions": 64,
              "n_window": 4,
              "n_window_infer": 16,
              "downsample_hidden_size": 4,
              "output_dim": \(outputDim),
              "activation_function": "gelu",
              "scale_embedding": false
            },
            "vision_config": {
              "model_type": "qwen3_omni_moe_vision_encoder",
              "depth": 1,
              "hidden_size": 16,
              "intermediate_size": 32,
              "out_hidden_size": 32,
              "num_heads": 2,
              "patch_size": 2,
              "spatial_merge_size": 2,
              "temporal_patch_size": 1,
              "num_position_embeddings": 16,
              "in_channels": 3,
              "hidden_act": "gelu_pytorch_tanh",
              "deepstack_visual_indexes": []
            },
            "text_config": {
              "model_type": "qwen3_omni_moe_text",
              "hidden_size": 32,
              "intermediate_size": 64,
              "num_hidden_layers": 1,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "head_dim": 16,
              "num_experts": 4,
              "num_experts_per_tok": 2,
              "moe_intermediate_size": 16,
              "decoder_sparse_step": 1,
              "norm_topk_prob": false,
              "rms_norm_eps": 1e-6,
              "rope_theta": 10000,
              "max_position_embeddings": 128,
              "tie_word_embeddings": false,
              "vocab_size": 64,
              "rope_scaling": {"type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": true}
            }
            """
        let body = ids.isEmpty ? blocks : ids + "\n" + blocks
        if flat {
            return "{\n\"model_type\": \"qwen3_omni_moe\",\n" + body + "\n}"
        }
        return "{\n\"model_type\": \"qwen3_omni_moe\",\n\"thinker_config\": {\n" + body + "\n}\n}"
    }
}

final class SharpAIQwen3OmniMoeConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIQwen3OmniMoeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIQwen3OmniMoeConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesThinkerConfigUnderWrapper() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny())

        XCTAssertEqual(config.modelType, "qwen3_omni_moe")
        XCTAssertEqual(config.audioTokenId, 30)
        XCTAssertEqual(config.imageTokenId, 31)
        XCTAssertEqual(config.videoTokenId, 32)
        XCTAssertEqual(config.visionStartTokenId, 33)
        XCTAssertEqual(config.audioConfiguration.dModel, 16)
        XCTAssertEqual(config.audioConfiguration.nWindowInfer, 16)
        XCTAssertEqual(config.audioConfiguration.outputDim, 32)
        XCTAssertEqual(config.textConfiguration.numExperts, 4)
        XCTAssertEqual(config.textConfiguration.numExpertsPerTok, 2)
        XCTAssertEqual(config.visionConfiguration.spatialMergeSize, 2)
    }

    func testDecodesFlatThinkerConfig() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny(flat: true))

        XCTAssertEqual(config.audioTokenId, 30)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.audioConfiguration.encoderLayers, 1)
    }

    func testMissingTokenIdsUseQwen3OmniDefaults() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny(tokenIds: false))

        XCTAssertEqual(config.audioTokenId, 151_646)
        XCTAssertEqual(config.imageTokenId, 151_655)
        XCTAssertEqual(config.videoTokenId, 151_656)
        XCTAssertEqual(config.visionStartTokenId, 151_652)
    }

    func testTinyConfigValidates() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny())
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsAudioOutputDimMismatch() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny(outputDim: 64))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testLanguageConfigurationCarriesTextAndTokenIds() throws {
        let config = try decode(SharpAIQwen3OmniMoeTestConfig.tiny())
        let language = try config.languageConfiguration()

        XCTAssertEqual(language.textConfiguration.numExperts, 4)
        XCTAssertEqual(language.textConfiguration.numHiddenLayers, 1)
        XCTAssertEqual(language.visionConfiguration.spatialMergeSize, 2)
        XCTAssertEqual(language.imageTokenIndex, 31)
        XCTAssertEqual(language.videoTokenIndex, 32)
        XCTAssertEqual(language.visionStartTokenId, 33)
    }
}
