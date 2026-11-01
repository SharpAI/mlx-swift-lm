// SharpAI fork overlay tests for llava_onevision2 configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAILlavaOnevision2ConfigurationTests: XCTestCase {

    // Shaped like the LLaVA-OneVision-2 8B Instruct fixture, with layer_types trimmed.
    private let fixtureJSON = """
        {
            "architectures": ["LlavaOnevision2ForConditionalGeneration"],
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "image_token_id": 151655,
            "model_type": "llava_onevision2",
            "text_config": {
                "model_type": "qwen3",
                "attention_bias": false,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "intermediate_size": 12288,
                "max_position_embeddings": 262144,
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "rope_theta": 8000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "vocab_size": 151936
            },
            "tie_word_embeddings": false,
            "video_token_id": 151656,
            "vision_config": {
                "attention_dropout": 0.0,
                "frame_windows_size": 4,
                "hidden_act": "gelu",
                "hidden_size": 1024,
                "image_size": 448,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-06,
                "layer_norm_type": "layer_norm",
                "max_position_embeddings": 8192,
                "model_type": "onevision_encoder",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 24,
                "out_hidden_size": 4096,
                "patch_size": 14,
                "rope_theta": 10000.0,
                "spatial_merge_size": 2,
                "text_hidden_size": 4096,
                "tokens_per_second": 1,
                "use_head": false
            },
            "vision_end_token_id": 151653,
            "vision_start_token_id": 151652
        }
        """

    private func decode(_ json: String) throws -> SharpAILlavaOnevision2Configuration {
        try JSONDecoder.json5().decode(
            SharpAILlavaOnevision2Configuration.self, from: Data(json.utf8))
    }

    func testDecodesTopLevelAndTokenIds() throws {
        let configuration = try decode(fixtureJSON)

        XCTAssertEqual(configuration.modelType, "llava_onevision2")
        XCTAssertEqual(configuration.architectures, ["LlavaOnevision2ForConditionalGeneration"])
        XCTAssertEqual(configuration.imageTokenId, 151655)
        XCTAssertEqual(configuration.videoTokenId, 151656)
        XCTAssertEqual(configuration.visionStartTokenId, 151652)
        XCTAssertEqual(configuration.visionEndTokenId, 151653)
    }

    func testDecodesTextConfigWithNestedRopeTheta() throws {
        let text = try decode(fixtureJSON).textConfiguration

        XCTAssertEqual(text.modelType, "qwen3")
        XCTAssertEqual(text.hiddenSize, 4096)
        XCTAssertEqual(text.hiddenLayers, 36)
        XCTAssertEqual(text.attentionHeads, 32)
        XCTAssertEqual(text.kvHeads, 8)
        XCTAssertEqual(text.headDim, 128)
        XCTAssertEqual(text.vocabularySize, 151936)
        XCTAssertEqual(text.ropeTheta, 8_000_000)
        XCTAssertFalse(text.tieWordEmbeddings)
    }

    func testDecodesVisionConfigFields() throws {
        let vision = try decode(fixtureJSON).visionConfiguration

        XCTAssertEqual(vision.modelType, "onevision_encoder")
        XCTAssertEqual(vision.hiddenSize, 1024)
        XCTAssertEqual(vision.hiddenLayers, 24)
        XCTAssertEqual(vision.attentionHeads, 16)
        XCTAssertEqual(vision.imageSize, 448)
        XCTAssertEqual(vision.patchSize, 14)
        XCTAssertEqual(vision.spatialMergeSize, 2)
        XCTAssertEqual(vision.outHiddenSize, 4096)
        XCTAssertEqual(vision.frameWindowSize, 4)
        XCTAssertEqual(vision.tokensPerSecond, 1)
        XCTAssertEqual(vision.hiddenAct, "gelu")
        XCTAssertFalse(vision.useHead)
    }

    func testDerivedImageTokenCount() throws {
        let configuration = try decode(fixtureJSON)

        XCTAssertEqual(configuration.visionGridSide, 32)
        XCTAssertEqual(configuration.imageTokensPerImage, 256)
    }

    func testFixtureConfigurationValidates() throws {
        XCTAssertNoThrow(try decode(fixtureJSON).validateModelConfiguration())
    }

    func testHeadDimDefaultsToHiddenOverHeadsWhenMissing() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"head_dim\": 128,", with: "")
        let text = try decode(json).textConfiguration

        XCTAssertEqual(text.headDim, 128)
    }

    func testAcceptsFlatRopeThetaInTextConfig() throws {
        // Renaming the nested block makes rope_theta a flat text_config key.
        let json = fixtureJSON.replacingOccurrences(
            of: "\"rope_parameters\": {", with: "\"rope_theta\": 1000000, \"unused\": {")
        let configuration = try decode(json)

        XCTAssertEqual(configuration.textConfiguration.ropeTheta, 1_000_000)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingRopeThetaFailsDecoding() {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"rope_theta\": 8000000,", with: "")

        XCTAssertThrowsError(try decode(json))
    }

    func testRejectsWrongModelType() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"model_type\": \"llava_onevision2\",", with: "\"model_type\": \"llava\",")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testRejectsHeadCountNotDivisibleByKVHeads() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 8", with: "\"num_key_value_heads\": 3")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testRejectsProjectorWidthMismatch() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"out_hidden_size\": 4096", with: "\"out_hidden_size\": 2048")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testRejectsImageSizeNotMultipleOfPatch() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"image_size\": 448", with: "\"image_size\": 450")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration())
    }

    func testRejectsUnsupportedVisionNormAndActivation() throws {
        let norm = fixtureJSON.replacingOccurrences(
            of: "\"layer_norm_type\": \"layer_norm\"", with: "\"layer_norm_type\": \"rms_norm\"")
        XCTAssertThrowsError(try decode(norm).validateModelConfiguration())

        let act = fixtureJSON.replacingOccurrences(
            of: "\"hidden_act\": \"gelu\"", with: "\"hidden_act\": \"quick_gelu\"")
        XCTAssertThrowsError(try decode(act).validateModelConfiguration())
    }

    func testRejectsNonPositiveVisionChannels() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"num_channels\": 3", with: "\"num_channels\": 0")
        XCTAssertThrowsError(try decode(json).validateModelConfiguration())
    }
}
