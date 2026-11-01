// SharpAI fork overlay tests for the muse_glimmer_assistant config and target check.
// Not part of upstream. Offline only: inline JSON, no checkpoint.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIMuseGlimmerAssistantConfigurationTests: XCTestCase {

    /// Copied from scripts/conformance/offline/fixtures/muse_glimmer_assistant/
    /// meta-models__Muse-Glimmer-30B-assistant.json.
    private let fixtureJSON = """
        {
          "architectures": ["MuseGlimmerAssistantModel"],
          "attention_dropout": 0,
          "block_size": 16,
          "bos_token_id": 200000,
          "dtype": "bfloat16",
          "eos_token_id": 200001,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 6656,
          "intermediate_size": 19968,
          "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
          "mask_token_id": 201818,
          "max_position_embeddings": 131072,
          "model_type": "muse_glimmer_assistant",
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "pad_token_id": 200018,
          "rms_norm_eps": 1e-05,
          "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
          "sliding_window": 2048,
          "target_layer_ids": [1, 13, 25, 37, 49],
          "transformers_version": "5.15.0.dev0"
        }
        """

    private func decode(_ json: String) throws -> SharpAIMuseGlimmerAssistantConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIMuseGlimmerAssistantConfiguration.self, from: Data(json.utf8))
    }

    private func replacing(_ old: String, with new: String) -> String {
        fixtureJSON.replacingOccurrences(of: old, with: new)
    }

    func testFixtureDecodesDimensions() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.modelType, "muse_glimmer_assistant")
        XCTAssertEqual(config.hiddenSize, 6656)
        XCTAssertEqual(config.intermediateSize, 19968)
        XCTAssertEqual(config.hiddenLayers, 5)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.slidingWindow, 2048)
        XCTAssertEqual(config.layerTypes.count, 5)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertEqual(config.blockSize, 16)
    }

    func testFixtureDecodesTopLevelTokenIds() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.bosTokenId, 200_000)
        XCTAssertEqual(config.eosTokenId, 200_001)
        XCTAssertEqual(config.padTokenId, 200_018)
        XCTAssertEqual(config.maskTokenId, 201_818)
    }

    func testFixtureDecodesTopLevelTargetLayerIds() throws {
        let config = try decode(fixtureJSON)

        XCTAssertEqual(config.targetLayerIds, [1, 13, 25, 37, 49])
    }

    func testMissingTopLevelIdsDecodeAsNil() throws {
        let json = """
            {
              "model_type": "muse_glimmer_assistant",
              "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
              "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
              "layer_types": ["full_attention"], "target_layer_ids": [0]
            }
            """
        let config = try decode(json)

        XCTAssertNil(config.bosTokenId)
        XCTAssertNil(config.eosTokenId)
        XCTAssertNil(config.padTokenId)
        XCTAssertNil(config.maskTokenId)
        XCTAssertNil(config.blockSize)
        XCTAssertNil(config.slidingWindow)
        XCTAssertEqual(config.targetLayerIds, [0])
        XCTAssertFalse(config.attentionBias)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testFixturePassesValidation() throws {
        let config = try decode(fixtureJSON)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testEmptyTargetLayerIdsFailValidation() throws {
        let config = try decode(replacing("[1, 13, 25, 37, 49]", with: "[]"))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testDuplicateTargetLayerIdsFailValidation() throws {
        let config = try decode(replacing("[1, 13, 25, 37, 49]", with: "[1, 1]"))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testLayerTypeCountMismatchFailsValidation() throws {
        let config = try decode(
            replacing(
                "\"layer_types\": [\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"sliding_attention\"]",
                with: "\"layer_types\": [\"sliding_attention\"]"))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testSlidingLayersWithoutWindowFailValidation() throws {
        let config = try decode(replacing("\"sliding_window\": 2048,", with: ""))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testWrongModelTypeFailsValidation() throws {
        let config = try decode(
            replacing("\"model_type\": \"muse_glimmer_assistant\"", with: "\"model_type\": \"llama\""))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    // MARK: - Target conformance

    func testFixtureIsCompatibleWithMuseGlimmerTargetShape() throws {
        let config = try decode(fixtureJSON)

        XCTAssertNoThrow(
            try config.validateTarget(
                modelType: "muse_glimmer", hiddenSize: 6656, hiddenLayers: 52))
    }

    func testWrongTargetModelTypeIsRejected() throws {
        let config = try decode(fixtureJSON)

        XCTAssertThrowsError(
            try config.validateTarget(
                modelType: "gemma4", hiddenSize: 6656, hiddenLayers: 52)
        ) { error in
            guard case SharpAIMuseGlimmerAssistantError.incompatibleTarget = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testHiddenSizeMismatchIsRejected() throws {
        let config = try decode(fixtureJSON)

        XCTAssertThrowsError(
            try config.validateTarget(
                modelType: "muse_glimmer", hiddenSize: 6144, hiddenLayers: 52))
    }

    func testTargetLayerIdBeyondTargetDepthIsRejected() throws {
        let config = try decode(fixtureJSON)

        // Layer 49 is the highest id, so a 49-layer target is too shallow.
        let shallowTarget = try makeTarget(hiddenLayers: 49)
        XCTAssertFalse(config.isCompatible(with: shallowTarget))

        let fullTarget = try makeTarget(hiddenLayers: 52)
        XCTAssertTrue(config.isCompatible(with: fullTarget))
    }

    /// Builds a text-only MuseGlimmer target config. Vision config uses its defaults.
    private func makeTarget(hiddenLayers: Int) throws -> MuseGlimmerConfiguration {
        let layerTypes = Array(repeating: "\"sliding_attention\"", count: hiddenLayers)
        let layerRope = Array(repeating: "500000.0", count: hiddenLayers)
        let json = """
            {
              "architectures": ["MuseGlimmerForConditionalGeneration"],
              "model_type": "muse_glimmer",
              "image_token_id": 200092,
              "video_token_id": 200093,
              "out_hidden_size": 6144,
              "projector_hidden_size": 4096,
              "vision_config": {},
              "text_config": {
                "model_type": "muse_glimmer_text",
                "vocab_size": 202048,
                "hidden_size": 6656,
                "intermediate_size": 19968,
                "num_hidden_layers": \(hiddenLayers),
                "num_attention_heads": 32,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "max_position_embeddings": 131072,
                "rms_norm_eps": 1e-05,
                "post_norm_eps": 1e-08,
                "attention_bias": false,
                "sliding_window": 2048,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611613513818404,
                "final_logit_softcapping": 20.0,
                "tie_word_embeddings": false,
                "layer_types": \(jsonArray(layerTypes)),
                "layer_rope_theta": \(jsonArray(layerRope))
              }
            }
            """
        return try JSONDecoder.json5().decode(
            MuseGlimmerConfiguration.self, from: Data(json.utf8))
    }

    /// Joins pre-rendered JSON scalars into a JSON array.
    private func jsonArray(_ values: [String]) -> String {
        "[" + values.joined(separator: ", ") + "]"
    }
}
