// SharpAI fork overlay tests for the muse_spark_dspark configuration. Not part of upstream.
// Offline only: inline JSON, no checkpoint.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMuseSparkDsparkConfigurationTests: XCTestCase {

    private let tinyJSON = """
        {
          "model_type": "muse_spark_dspark",
          "architectures": ["MuseSparkDSparkModel"],
          "hidden_size": 16, "ffn_dim_multiplier": 1.5,
          "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
          "head_dim": 8, "vocab_size": 50, "draft_vocab_size": 40,
          "moe": {"is_enabled": false},
          "global_attn_cfg": [4, 0],
          "target_layer_ids": [1, 3], "num_target_layers": 2, "target_hidden_size": 6,
          "markov_rank": 4, "enable_confidence_head": true, "confidence_head_with_markov": true,
          "use_attn_o_gate": true, "use_attn_o_norm": true, "use_residual_gate": true,
          "draft_position_mode": "rope"
        }
        """

    private func decode(_ json: String) throws -> SharpAIMuseSparkDsparkConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIMuseSparkDsparkConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesTinyConfigAndValidates() throws {
        let config = try decode(tinyJSON)

        XCTAssertEqual(config.modelType, "muse_spark_dspark")
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSizeMlp, 24)
        XCTAssertEqual(config.targetLayerIds, [1, 3])
        XCTAssertEqual(config.markovRank, 4)
        XCTAssertTrue(config.useRope)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDraftPositionModeOverridesUseRope() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"draft_position_mode\": \"rope\"", with: "\"draft_position_mode\": \"nope\"")
        let config = try decode(json)

        XCTAssertFalse(config.useRope)
        XCTAssertFalse(config.layerUsesRope(0))
    }

    func testSlidingWindowFollowsBackwardLayerCount() throws {
        let config = try decode(tinyJSON)

        XCTAssertEqual(config.slidingWindow(forLayer: 0), 4)
        XCTAssertNil(config.slidingWindow(forLayer: 1))
    }

    func testMetapMupOutputMultiplier() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"markov_rank\": 4,",
            with: "\"markov_rank\": 4, \"metap\": {\"use_metap\": true, \"base_width\": 8},")
        let config = try decode(json)

        XCTAssertEqual(config.outputMultiplier, 0.5, accuracy: 1e-6)
    }

    func testMoEDefaultsOnAndIsRejected() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"moe\": {\"is_enabled\": false},", with: "")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration(let detail) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertTrue(detail.contains("MoE layers"))
        }
    }

    func testUnknownResidualDtypeIsRejected() throws {
        let json = tinyJSON.replacingOccurrences(
            of: "\"draft_position_mode\": \"rope\"",
            with: "\"draft_position_mode\": \"rope\", \"residual_dtype\": \"int8\"")
        let config = try decode(json)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testMissingMarkovRankFailsDecode() {
        let json = tinyJSON.replacingOccurrences(of: "\"markov_rank\": 4,", with: "")

        XCTAssertThrowsError(try decode(json))
    }

    func testVLLMFixtureDecodesAndValidates() throws {
        // Fixture from vLLM's tests/transformers_utils/test_muse_spark_config.py.
        let fixture = """
            {
              "model_type": "muse_spark_dspark",
              "num_hidden_layers": 2,
              "moe": {"is_enabled": false},
              "global_attn_cfg": [2048],
              "layer_types": ["sliding_attention", "sliding_attention"],
              "target_layer_ids": [1, 3],
              "num_target_layers": 2,
              "markov_rank": 8,
              "markov_head_type": "vanilla",
              "dflash_config": {"causal": true, "target_layer_ids": [1, 3]}
            }
            """
        let config = try decode(fixture)

        XCTAssertEqual(config.numHiddenLayers, 2)
        XCTAssertEqual(config.globalAttnSlidingWindows, [2048])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
