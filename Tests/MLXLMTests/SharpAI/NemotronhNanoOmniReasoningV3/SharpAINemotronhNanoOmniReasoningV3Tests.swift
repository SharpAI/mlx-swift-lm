// SharpAI fork overlay tests: NemotronH_Nano_Omni_Reasoning_V3. Not part of upstream mlx-swift-lm.
// Written without compiling or running. No network and no real weights.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemotronhNanoOmniReasoningV3Tests: XCTestCase {

    // Tiny shapes with the checkpoint's key layout. The RADIO tower is ViT-H depth and heads with
    // width 64, so it stays cheap to build. vit_hidden_size 64 gives mlp1 an input of 256.
    private let tinyConfigJSON = """
        {
            "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
            "architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
            "llm_config": {
                "model_type": "nemotron_h",
                "vocab_size": 100,
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 16,
                "ssm_state_size": 16,
                "conv_kernel": 4,
                "n_groups": 2,
                "intermediate_size": 128,
                "moe_intermediate_size": 64,
                "moe_shared_expert_intermediate_size": 64,
                "n_routed_experts": 4,
                "num_experts_per_tok": 2,
                "hybrid_override_pattern": "M*-E",
                "n_group": 2,
                "topk_group": 1
            },
            "vit_hidden_size": 64,
            "projector_hidden_size": 32,
            "downsample_ratio": 0.5,
            "ps_version": "v2",
            "vision_config": {
                "model_type": "radio",
                "patch_size": 16,
                "args": {
                    "model": "vit_huge_patch16_224",
                    "cpe_max_size": 64,
                    "cls_token_per_teacher": true,
                    "register_multiple": 10,
                    "teachers": [{"name": "clip"}, {"name": "siglip"}, {"name": "dino_v2"}, {"name": "sam"}]
                }
            },
            "sound_config": null
        }
        """

    private let soundConfigJSON = """
        {"model_type": "parakeet", "hidden_size": 16, "projection_hidden_size": 32, "projection_bias": false}
        """

    private func decodeConfig(_ json: String) throws -> SharpAINemotronhNanoOmniReasoningV3Configuration {
        try JSONDecoder().decode(
            SharpAINemotronhNanoOmniReasoningV3Configuration.self,
            from: Data(json.utf8))
    }

    // MARK: - Configuration

    func testConfigurationDecodesNestedLLMConfig() throws {
        let config = try decodeConfig(tinyConfigJSON)

        XCTAssertEqual(config.modelType, "NemotronH_Nano_Omni_Reasoning_V3")
        XCTAssertEqual(config.architectures, ["NemotronH_Nano_Omni_Reasoning_V3"])
        XCTAssertEqual(config.llmConfig.vocabSize, 100)
        XCTAssertEqual(config.llmConfig.hiddenSize, 64)
        XCTAssertEqual(config.llmConfig.hybridOverridePattern, "M*-E")
        XCTAssertEqual(config.llmConfig.numHiddenLayers, 4)
    }

    func testConfigurationKeepsVisionConfigAndNullSoundConfig() throws {
        let config = try decodeConfig(tinyConfigJSON)

        XCTAssertNotNil(config.visionConfig)
        XCTAssertNil(config.soundConfig)
        XCTAssertNil(config.soundProjection)
    }

    func testConfigurationDecodesRADIOTowerShape() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let tower = try XCTUnwrap(config.visionTower)

        XCTAssertEqual(tower.patchSize, 16)
        XCTAssertEqual(tower.embedDim, 64)
        XCTAssertEqual(tower.depth, 32)
        XCTAssertEqual(tower.numHeads, 16)
        XCTAssertEqual(tower.mlpHiddenSize, 256)
        XCTAssertEqual(tower.cpeMaxSize, 64)
        // Four teachers give four cls tokens. Registers pad the total to a multiple of 10.
        XCTAssertEqual(tower.numClsTokens, 4)
        XCTAssertEqual(tower.numSkipTokens, 10)
    }

    func testConfigurationRejectsUnportedVisionModel() {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "vit_huge_patch16_224", with: "vit_large_patch16_224")

        XCTAssertThrowsError(try decodeConfig(json)) { error in
            XCTAssertEqual(
                error as? SharpAINemotronhNanoOmniReasoningV3Error,
                .unsupportedVisionModel("vit_large_patch16_224"))
        }
    }

    func testConfigurationDecodesSoundProjectionShape() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"sound_config\": null", with: "\"sound_config\": \(soundConfigJSON)")
        let config = try decodeConfig(json)

        XCTAssertEqual(
            config.soundProjection,
            SharpAINemotronhNanoOmniReasoningV3SoundProjectionShape(
                inputSize: 16, hiddenSize: 32, bias: false))
    }

    func testConfigurationRequiresLLMConfig() {
        let json = """
            {
                "model_type": "NemotronH_Nano_Omni_Reasoning_V3",
                "vision_config": {}
            }
            """
        XCTAssertThrowsError(try decodeConfig(json))
    }

    func testConfigurationValidationRejectsEmptyPattern() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"M*-E\"", with: "\"\"")
        let config = try decodeConfig(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testConfigurationValidationRejectsUnportedDownsampleRatio() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"downsample_ratio\": 0.5", with: "\"downsample_ratio\": 0.25")
        let config = try decodeConfig(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testConfigurationValidationAcceptsTinyConfig() throws {
        let config = try decodeConfig(tinyConfigJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testSuperArchitectureStringDecodesToSameModelType() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "NemotronH_Nano_Omni_Reasoning_V3\"]",
            with: "NemotronH_Super_Omni_Reasoning_V3\"]")
        let config = try decodeConfig(json)

        XCTAssertEqual(config.architectures, ["NemotronH_Super_Omni_Reasoning_V3"])
        XCTAssertEqual(config.modelType, "NemotronH_Nano_Omni_Reasoning_V3")
    }

    // MARK: - Sanitize

    func testSanitizeKeepsLanguageAndBuiltTowerKeysAndDropsTheRest() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        let weights: [String: MLXArray] = [
            "language_model.backbone.embeddings.weight": MLXArray.ones([100, 64]),
            "vision_model.radio_model.model.blocks.0.mlp.fc2.weight": MLXArray.ones([2, 2]),
            "vision_model.radio_model.model.patch_generator.video_embedder.weight": MLXArray.ones([2, 2]),
            "vision_model.radio_model.input_conditioner.norm_mean": MLXArray.ones([3, 1, 1]),
            "vision_model.radio_model.input_conditioner.norm_std": MLXArray.ones([3, 1, 1]),
            "vision_model.radio_model.summary_idxs": MLXArray.ones([2]),
            "mlp1.3.weight": MLXArray.ones([2, 2]),
            "sound_encoder.encoder.subsampling.layers.2.weight": MLXArray.ones([2, 2]),
            "vision_tower.weight": MLXArray.ones([2, 2]),
            "mtp.layers.0.weight": MLXArray.ones([2, 2]),
        ]

        let sanitized = model.sanitize(weights: weights)

        // sound_projection is not in this config (sound_config is null), so it is not kept.
        XCTAssertEqual(
            Set(sanitized.keys),
            [
                "language_model.backbone.embeddings.weight",
                "vision_model.radio_model.model.blocks.0.mlp.fc2.weight",
                "mlp1.3.weight",
            ])
    }

    func testPrepareRejectsImageInput() throws {
        let model = SharpAINemotronhNanoOmniReasoningV3Model(try decodeConfig(tinyConfigJSON))
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 2, 3])[.newAxis, .ellipsis]),
            image: LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 3, 64, 64])))
        let cache = try model.newCache(parameters: nil)

        XCTAssertThrowsError(
            try model.prepare(input, cache: cache, state: nil, prefill: PrefillParameters())
        ) { error in
            XCTAssertEqual(
                error as? SharpAINemotronhNanoOmniReasoningV3Error, .multimodalInputNotSupported)
        }
    }

    func testSanitizeKeepsSoundProjectionWhenBuilt() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"sound_config\": null", with: "\"sound_config\": \(soundConfigJSON)")
        let model = SharpAINemotronhNanoOmniReasoningV3Model(try decodeConfig(json))

        let weights: [String: MLXArray] = [
            "sound_projection.linear1.weight": MLXArray.ones([32, 16]),
            "sound_encoder.encoder.layers.0.weight": MLXArray.ones([2, 2]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), ["sound_projection.linear1.weight"])
    }

    func testSanitizeAddsLanguagePrefixToBareTextKeys() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        let weights: [String: MLXArray] = [
            "backbone.embeddings.weight": MLXArray.ones([100, 64])
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(Set(sanitized.keys), ["language_model.backbone.embeddings.weight"])
    }

    func testSanitizeStacksExpertsUnderLanguagePrefix() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        // Layer 3 is the "E" block in "M*-E". Expert count is 4 and moe_intermediate_size is 64.
        var weights = [String: MLXArray]()
        for e in 0 ..< 4 {
            weights["language_model.backbone.layers.3.mixer.experts.\(e).up_proj.weight"] =
                MLXArray.ones([64, 64])
            weights["language_model.backbone.layers.3.mixer.experts.\(e).down_proj.weight"] =
                MLXArray.ones([64, 64])
        }

        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["language_model.backbone.layers.3.mixer.switch_mlp.fc1.weight"]?.shape,
            [4, 64, 64])
        XCTAssertEqual(
            sanitized["language_model.backbone.layers.3.mixer.switch_mlp.fc2.weight"]?.shape,
            [4, 64, 64])
        XCTAssertNil(sanitized["language_model.backbone.layers.3.mixer.experts.0.up_proj.weight"])
    }

    // MARK: - Model

    func testForwardPassWithRandomWeights() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        let input = MLXArray([1, 2, 3, 4, 5])[.newAxis, .ellipsis]
        let output = model.callAsFunction(input, cache: nil)

        XCTAssertEqual(output.shape, [1, 5, 100])
    }

    func testNewCacheHasOneEntryPerMambaOrAttentionBlock() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        // "M*-E" has one Mamba and one attention block, so two caches.
        let caches = try model.newCache(parameters: nil)
        XCTAssertEqual(caches.count, 2)
    }

    func testImageFeaturesProjectToLLMWidthWithPixelShuffle() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        // 64x64 pixels with patch 16 give a 4x4 patch grid. The 2x pixel shuffle leaves 2x2 = 4
        // tokens, each 4 * 64 = 256 wide, which mlp1 maps to the llm width of 64.
        let pixels = MLXArray.zeros([1, 3, 64, 64])
        let features = try model.imageFeatures(pixels)

        XCTAssertEqual(features.shape, [1, 4, 64])
    }

    func testImageFeaturesRejectsOddPatchGrid() throws {
        let config = try decodeConfig(tinyConfigJSON)
        let model = SharpAINemotronhNanoOmniReasoningV3Model(config)

        // 48x48 with patch 16 gives a 3x3 grid, which the 2x pixel shuffle cannot tile.
        XCTAssertThrowsError(try model.imageFeatures(MLXArray.zeros([1, 3, 48, 48])))
    }

    // MARK: - Registration

    func testRegistryCreatesModelFromModelType() throws {
        let creator = try XCTUnwrap(
            SharpAINemotronhNanoOmniReasoningV3Overlay.creators[
                "NemotronH_Nano_Omni_Reasoning_V3"])

        let model = try creator(Data(tinyConfigJSON.utf8))

        XCTAssertTrue(model is SharpAINemotronhNanoOmniReasoningV3Model)
    }

    func testRegistryRejectsInvalidConfiguration() throws {
        let creator = try XCTUnwrap(
            SharpAINemotronhNanoOmniReasoningV3Overlay.creators[
                "NemotronH_Nano_Omni_Reasoning_V3"])
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"M*-E\"", with: "\"\"")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    // MARK: - Towers and projectors

    func testVisionTowerAndProjectorKeysAreBuilt() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"sound_config\": null", with: "\"sound_config\": \(soundConfigJSON)")
        let model = SharpAINemotronhNanoOmniReasoningV3Model(try decodeConfig(json))
        let parameterKeys = model.parameters().flattened().map(\.0)

        XCTAssertTrue(parameterKeys.contains("vision_model.radio_model.model.blocks.0.mlp.fc2.weight"))
        XCTAssertTrue(parameterKeys.contains("vision_model.radio_model.model.patch_generator.pos_embed"))
        // mlp1 is nn.Sequential(RMSNorm, Linear, SquaredReLU, Linear): indices 0, 1 and 3.
        XCTAssertTrue(parameterKeys.contains("mlp1.0.weight"))
        XCTAssertTrue(parameterKeys.contains("mlp1.1.weight"))
        XCTAssertTrue(parameterKeys.contains("mlp1.3.weight"))
        XCTAssertTrue(parameterKeys.contains("sound_projection.norm.weight"))
        XCTAssertTrue(parameterKeys.contains("sound_projection.linear1.weight"))
        XCTAssertTrue(parameterKeys.contains("sound_projection.linear2.weight"))
    }

    // MARK: - Known gaps

    /// Known gap: the Parakeet sound encoder (`sound_encoder.encoder.*`) is not ported. The
    /// projection in front of the LLM is built (see testVisionTowerAndProjectorKeysAreBuilt).
    /// Source searched: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16/raw/main/audio_model.py
    /// only wraps `ParakeetEncoder` from transformers. The FastConformer layer structure (subsampling
    /// convs, rel-pos attention, conv module) is not in the fetched sources, so no Swift port exists.
    /// It flips to a hard failure once the encoder keys appear in parameters().
    func testParakeetSoundEncoderNotImplemented() throws {
        let json = tinyConfigJSON.replacingOccurrences(
            of: "\"sound_config\": null", with: "\"sound_config\": \(soundConfigJSON)")
        let model = SharpAINemotronhNanoOmniReasoningV3Model(try decodeConfig(json))
        let parameterKeys = model.parameters().flattened().map(\.0)

        XCTExpectFailure(
            "KNOWN GAP: Parakeet sound encoder (sound_encoder.encoder.*) is not ported; the FastConformer layers are not in the fetched sources (audio_model.py wraps transformers' ParakeetEncoder)"
        ) {
            XCTAssertTrue(parameterKeys.contains { $0.hasPrefix("sound_encoder.") })
        }
    }
}
