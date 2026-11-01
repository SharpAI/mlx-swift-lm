// SharpAI fork overlay tests for granite_swa config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIGraniteSwaConfigurationTests: XCTestCase {

    private let fullJSON = """
        {
            "model_type": "granite_swa",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "rope_theta": 50000.0,
            "max_position_embeddings": 4096,
            "attention_bias": true,
            "mlp_bias": true,
            "tie_word_embeddings": false,
            "sliding_window": 1024,
            "logits_scaling": 2.0,
            "attention_multiplier": 0.5,
            "embedding_multiplier": 3.0,
            "residual_multiplier": 0.25
        }
        """

    func testDecodesAllFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(fullJSON.utf8))

        XCTAssertEqual(config.modelType, "granite_swa")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.ropeTheta, 50_000.0)
        XCTAssertEqual(config.maxPositionEmbeddings, 4096)
        XCTAssertTrue(config.attentionBias)
        XCTAssertTrue(config.mlpBias)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.slidingWindow, 1024)
        XCTAssertEqual(config.logitsScaling, 2.0)
        XCTAssertEqual(config.attentionMultiplier, 0.5)
        XCTAssertEqual(config.embeddingMultiplier, 3.0)
        XCTAssertEqual(config.residualMultiplier, 0.25)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseTransformersDefaults() throws {
        // Defaults from transformers GraniteSWAConfig and RotaryEmbeddingConfigMixin.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "granite_swa")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertNil(config.ropeScaling)
        XCTAssertEqual(config.maxPositionEmbeddings, 8192)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.slidingWindow, 128)
        XCTAssertEqual(config.layerTypes, ["full_attention", "sliding_attention"])
        XCTAssertEqual(config.layerRopeTheta, [10_000.0, 10_000.0])
        XCTAssertEqual(config.logitsScaling, 1.0)
        XCTAssertEqual(config.attentionMultiplier, 1.0)
        XCTAssertEqual(config.embeddingMultiplier, 1.0)
        XCTAssertEqual(config.residualMultiplier, 1.0)
    }

    func testDecodesPerLayerArrays() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 3,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
                "layer_rope_theta": [10000.0, 0, 10000.0]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(
            config.layerTypes, ["sliding_attention", "full_attention", "sliding_attention"])
        XCTAssertEqual(config.layerRopeTheta, [10_000.0, 0.0, 10_000.0])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingRequiredFieldThrows() {
        let json = """
            { "hidden_size": 64, "num_hidden_layers": 2 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIGraniteSwaConfiguration.self, from: Data(json.utf8)))
    }

    func testValidationRejectsHeadMismatch() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_key_value_heads": 3,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonPositiveSlidingWindow() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512,
                "sliding_window": 0
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIGraniteSwaConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
