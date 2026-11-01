// SharpAI fork overlay tests for the nemotron_h_omni configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemotronHOmniConfigurationTests: XCTestCase {

    private let fixtureJSON = """
        {
            "architectures": ["NemotronH_Omni_Reasoning_V3"],
            "downsample_ratio": 0.5,
            "force_image_size": 512,
            "model_type": "nemotron_h_omni",
            "patch_size": 16,
            "projector_hidden_size": 20480,
            "sound_config": null,
            "video_temporal_patch_size": 2,
            "vit_hidden_size": 1280,
            "vision_config": {"model_type": "radio", "hidden_size": 1280},
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 131072,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "mamba_num_heads": 128,
                "mamba_head_dim": 64,
                "ssm_state_size": 128,
                "conv_kernel": 4,
                "n_groups": 8,
                "intermediate_size": 2688,
                "moe_intermediate_size": 2688,
                "moe_shared_expert_intermediate_size": 5376,
                "n_routed_experts": 512,
                "num_experts_per_tok": 22,
                "moe_latent_size": 1024,
                "layers_block_type": ["mamba", "moe", "attention"]
            }
        }
        """

    private func decode(_ json: String) throws -> SharpAINemotronHOmniConfiguration {
        try JSONDecoder().decode(SharpAINemotronHOmniConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTopLevelFieldsAndNestedTextConfig() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "nemotron_h_omni")
        XCTAssertEqual(config.architectures, ["NemotronH_Omni_Reasoning_V3"])
        XCTAssertEqual(config.llmConfig.vocabSize, 131072)
        XCTAssertEqual(config.llmConfig.hybridOverridePattern, "ME*")
        XCTAssertEqual(config.llmConfig.moeLatentSize, 1024)
        XCTAssertEqual(config.downsampleRatio, 0.5)
        XCTAssertEqual(config.projectorHiddenSize, 20480)
        XCTAssertEqual(config.visionHiddenSize, 1280)
        XCTAssertEqual(config.patchSize, 16)
        XCTAssertEqual(config.forceImageSize, 512)
        XCTAssertEqual(config.videoTemporalPatchSize, 2)
    }

    func testKeepsVisionConfigRawAndNullSoundConfigAsNil() throws {
        let config = try decode(fixtureJSON)

        XCTAssertNotNil(config.visionConfig)
        XCTAssertNil(config.soundConfig)
    }

    func testFixtureConfigValidates() throws {
        try decode(fixtureJSON).validateModelConfiguration()
    }

    func testZeroLatentSizeFailsValidation() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"moe_latent_size\": 1024", with: "\"moe_latent_size\": 0")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAINemotronHOmniError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testEmptyLayerListFailsValidation() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "[\"mamba\", \"moe\", \"attention\"]", with: "[]")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAINemotronHOmniError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testMissingLLMConfigFailsDecoding() {
        XCTAssertThrowsError(try decode("{\"model_type\": \"nemotron_h_omni\"}"))
    }
}
