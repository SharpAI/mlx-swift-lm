// SharpAI fork overlay tests for exaone4_5 config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

@testable import MLXLLM

final class SharpAIExaone45ConfigurationTests: XCTestCase {

    private let fullJSON = """
        {
            "model_type": "exaone4_5",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "rms_norm_eps": 0.00001,
                "max_position_embeddings": 4096,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": true
            },
            "vision_config": {
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "out_hidden_size": 256
            }
        }
        """

    private let sparseJSON = """
        {
            "model_type": "exaone4_5",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "max_position_embeddings": 4096
            }
        }
        """

    func testDecodesNestedTextAndVisionConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: Data(fullJSON.utf8))

        XCTAssertEqual(config.modelType, "exaone4_5")
        XCTAssertEqual(config.textDimensions.vocabSize, 1024)
        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.hiddenLayers, 4)
        XCTAssertEqual(config.textDimensions.attentionHeads, 8)
        XCTAssertEqual(config.textDimensions.kvHeads, 2)
        XCTAssertEqual(config.textDimensions.headDim, 32)
        XCTAssertTrue(config.textDimensions.tieWordEmbeddings)

        let vision = try XCTUnwrap(config.visionDimensions)
        XCTAssertEqual(vision.hiddenSize, 128)
        XCTAssertEqual(vision.depth, 2)
        XCTAssertEqual(vision.kvHeads, 2)
        XCTAssertEqual(vision.patchSize, 16)
    }

    func testSparseTextConfigFillsDefaults() throws {
        // UNVERIFIED: default values follow the HF Exaone4 convention.
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: Data(sparseJSON.utf8))

        XCTAssertEqual(config.textDimensions.kvHeads, 8)
        XCTAssertEqual(config.textDimensions.headDim, 32)
        XCTAssertEqual(config.textDimensions.ropeTheta, 10_000)
        XCTAssertEqual(config.textDimensions.rmsNormEps, 1e-5)
        XCTAssertFalse(config.textDimensions.tieWordEmbeddings)
        XCTAssertNil(config.visionDimensions)
    }

    func testTextConfigConvertsToStrictExaone4Configuration() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: Data(fullJSON.utf8))

        let text = try config.exaone4TextConfiguration()

        XCTAssertEqual(text.hiddenSize, 256)
        XCTAssertEqual(text.hiddenLayers, 4)
        XCTAssertEqual(text.attentionHeads, 8)
        XCTAssertEqual(text.kvHeads, 2)
        XCTAssertEqual(text.headDim, 32)
        XCTAssertEqual(text.vocabularySize, 1024)
        XCTAssertTrue(text.tieWordEmbeddings)
    }

    func testSparseConfigConvertsToStrictExaone4Configuration() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: Data(sparseJSON.utf8))

        let text = try config.exaone4TextConfiguration()

        XCTAssertEqual(text.kvHeads, 8)
        XCTAssertEqual(text.headDim, 32)
        XCTAssertFalse(text.tieWordEmbeddings)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let flat = Data(
            """
            {
                "model_type": "exaone4_5",
                "vocab_size": 512,
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "num_attention_heads": 4,
                "max_position_embeddings": 2048
            }
            """.utf8)

        let config = try JSONDecoder.json5().decode(SharpAIExaone45Configuration.self, from: flat)

        XCTAssertEqual(config.textDimensions.vocabSize, 512)
        XCTAssertEqual(config.textDimensions.hiddenSize, 128)
        XCTAssertNil(config.visionDimensions)
    }

    func testMissingMaxPositionEmbeddingsThrows() {
        let missing = Data(
            """
            {
                "model_type": "exaone4_5",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 8
                }
            }
            """.utf8)

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(SharpAIExaone45Configuration.self, from: missing))
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: Data(fullJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    private func configJSON(textKV: Int, visionKV: Int) -> Data {
        Data(
            """
            {
                "model_type": "exaone4_5",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": 256,
                    "num_hidden_layers": 4,
                    "intermediate_size": 512,
                    "num_attention_heads": 8,
                    "num_key_value_heads": \(textKV),
                    "head_dim": 32,
                    "max_position_embeddings": 4096
                },
                "vision_config": {
                    "hidden_size": 128,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": \(visionKV)
                }
            }
            """.utf8)
    }

    func testTextHeadsNotDivisibleByKvHeadsFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: configJSON(textKV: 3, visionKV: 2))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testVisionHeadsNotDivisibleByKvHeadsFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIExaone45Configuration.self, from: configJSON(textKV: 2, visionKV: 3))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
