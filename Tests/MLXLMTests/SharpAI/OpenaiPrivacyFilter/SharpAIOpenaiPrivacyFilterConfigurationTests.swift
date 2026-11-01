// SharpAI fork overlay tests for the openai_privacy_filter configuration. Not part of upstream.

import Foundation
import MLX
import XCTest

@testable import MLXLLM

final class SharpAIOpenaiPrivacyFilterConfigurationTests: XCTestCase {

    private let checkpointJSON = """
        {
            "model_type": "openai_privacy_filter",
            "architectures": ["OpenAIPrivacyFilterForTokenClassification"],
            "num_hidden_layers": 8,
            "num_local_experts": 128,
            "num_experts_per_tok": 4,
            "vocab_size": 200064,
            "hidden_size": 640,
            "intermediate_size": 640,
            "head_dim": 64,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "sliding_window": 128,
            "rms_norm_eps": 1e-05,
            "attention_bias": true,
            "rope_parameters": {
                "rope_type": "yarn",
                "rope_theta": 150000.0,
                "factor": 32.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": false,
                "original_max_position_embeddings": 4096
            },
            "id2label": {"0": "O", "1": "B-secret", "2": "I-secret"}
        }
        """

    func testDecodesCheckpointFields() throws {
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(checkpointJSON.utf8))

        XCTAssertEqual(config.hiddenLayers, 8)
        XCTAssertEqual(config.localExperts, 128)
        XCTAssertEqual(config.expertsPerToken, 4)
        XCTAssertEqual(config.hiddenSize, 640)
        XCTAssertEqual(config.headDimensions, 64)
        XCTAssertEqual(config.attentionHeads, 14)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.slidingWindow, 128)
        XCTAssertEqual(config.rope.ropeType, "yarn")
        XCTAssertEqual(config.rope.factor, 32.0, accuracy: 1e-6)
        XCTAssertFalse(config.rope.truncate)
    }

    func testLabelCountComesFromId2Label() throws {
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(checkpointJSON.utf8))

        XCTAssertEqual(config.numLabels, 3)
    }

    func testMissingFieldsUseTransformersDefaults() throws {
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.numLabels, 33)
        XCTAssertEqual(config.headDimensions, 64)
        XCTAssertEqual(config.slidingWindow, 128)
        XCTAssertEqual(config.rope.originalMaxPositionEmbeddings, 4096)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testYarnAttentionFactorMatchesReference() throws {
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(checkpointJSON.utf8))

        // transformers: 0.1 * ln(factor) + 1 with factor 32.
        let expected = Float(0.1 * log(32.0) + 1.0)
        let factor = SharpAIOpenaiPrivacyFilterRope.yarn(config).attentionFactor
        XCTAssertEqual(factor, expected, accuracy: 1e-5)
    }

    func testRejectsMoreExpertsPerTokenThanExperts() throws {
        let invalid = checkpointJSON.replacingOccurrences(
            of: "\"num_experts_per_tok\": 4", with: "\"num_experts_per_tok\": 256")
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(invalid.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnsupportedRopeType() throws {
        let invalid = checkpointJSON.replacingOccurrences(
            of: "\"rope_type\": \"yarn\"", with: "\"rope_type\": \"linear\"")
        let config = try JSONDecoder().decode(
            SharpAIOpenaiPrivacyFilterConfiguration.self, from: Data(invalid.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
