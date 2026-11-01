// SharpAI fork overlay tests for muse_glimmer_text config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Shared inline fixtures. The tiny shape keeps the forward test cheap.
enum SharpAIMuseGlimmerTextTestFixtures {
    static let tinyJSON = """
        {
            "model_type": "muse_glimmer_text",
            "architectures": ["MuseGlimmerForCausalLM"],
            "vocab_size": 40,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "attention_bias": false,
            "sliding_window": 4,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.5,
            "final_logit_softcapping": 20.0,
            "tie_word_embeddings": false,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"}
        }
        """

    static func configuration(_ json: String = tinyJSON) throws
        -> SharpAIMuseGlimmerTextConfiguration
    {
        try JSONDecoder.json5().decode(
            SharpAIMuseGlimmerTextConfiguration.self, from: Data(json.utf8))
    }
}

final class SharpAIMuseGlimmerTextConfigurationTests: XCTestCase {

    func testDecodesCoreFields() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()

        XCTAssertEqual(config.modelType, "muse_glimmer_text")
        XCTAssertEqual(config.vocabularySize, 40)
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSize, 32)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.attentionHeads, 2)
        XCTAssertEqual(config.kvHeads, 1)
        XCTAssertEqual(config.headDimensions, 8)
        XCTAssertEqual(config.slidingWindow, 4)
        XCTAssertEqual(config.outputMultiplier, 0.5, accuracy: 1e-6)
        XCTAssertEqual(config.finalLogitSoftcapping, 20.0, accuracy: 1e-6)
        XCTAssertEqual(config.ropeTheta, 500_000.0, accuracy: 1)
    }

    func testDerivesLayerTypesWhenAbsent() throws {
        // Every 4th layer counting back from the last is full attention.
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()

        XCTAssertEqual(
            config.layerTypes,
            ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"])
    }

    func testDerivesNoPERopeThetaForFullLayers() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()

        XCTAssertEqual(config.layerRopeTheta.count, 4)
        XCTAssertEqual(config.layerRopeTheta[0], 500_000.0, accuracy: 1)
        XCTAssertEqual(config.layerRopeTheta[3], 0, accuracy: 1e-6)
    }

    func testUsesExplicitLayerTypesAndRopeTheta() throws {
        let json = """
            {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 2,
                "layer_types": ["full_attention", "sliding_attention"],
                "layer_rope_theta": [0, 10000.0]
            }
            """
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration(json)

        XCTAssertEqual(config.layerTypes, ["full_attention", "sliding_attention"])
        XCTAssertEqual(config.layerRopeTheta, [0, 10_000])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDefaultsWhenOptionalKeysAbsent() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration("{}")

        XCTAssertEqual(config.modelType, "muse_glimmer_text")
        XCTAssertEqual(config.vocabularySize, 202_048)
        XCTAssertEqual(config.hiddenLayers, 52)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDimensions, 128)
        XCTAssertEqual(config.slidingWindow, 2048)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testValidationRejectsLayerTypeCountMismatch() throws {
        let json = """
            {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 4,
                "layer_types": ["full_attention", "sliding_attention"]
            }
            """
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsUnknownLayerType() throws {
        let json = """
            {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 1,
                "layer_types": ["linear_attention"]
            }
            """
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration(json)

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
