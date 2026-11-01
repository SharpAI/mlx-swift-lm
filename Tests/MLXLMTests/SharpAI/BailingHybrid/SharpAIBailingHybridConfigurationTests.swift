// SharpAI fork overlay tests for bailing_hybrid config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIBailingHybridConfigurationTests: XCTestCase {

    // UNVERIFIED: key set is the bailing_moe base set plus an assumed nextn key.
    private let fullJSON = """
        {
            "model_type": "bailing_hybrid",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 600000.0,
            "tie_word_embeddings": true,
            "num_experts": 16,
            "num_experts_per_tok": 4,
            "moe_intermediate_size": 32,
            "num_shared_experts": 1,
            "first_k_dense_replace": 1,
            "num_nextn_predict_layers": 1
        }
        """

    func testDecodesAllFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(fullJSON.utf8))

        XCTAssertEqual(config.modelType, "bailing_hybrid")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.rmsNormEps, 1e-5, accuracy: 1e-9)
        XCTAssertEqual(config.ropeTheta, 600_000.0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.numExperts, 16)
        XCTAssertEqual(config.numExpertsPerTok, 4)
        XCTAssertEqual(config.moeIntermediateSize, 32)
        XCTAssertEqual(config.numSharedExperts, 1)
        XCTAssertEqual(config.firstKDenseReplace, 1)
        XCTAssertEqual(config.numNextnPredictLayers, 1)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        // UNVERIFIED: default values are guesses, pinned here so a change is visible.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "bailing_hybrid")
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNil(config.numExperts)
        XCTAssertEqual(config.numNextnPredictLayers, 0)
    }

    func testRingStyleConfigWithZeroNextnLayers() throws {
        // UNVERIFIED: a 0-nextn config is assumed, not taken from a real checkpoint.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "num_nextn_predict_layers": 0
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.numNextnPredictLayers, 0)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingRequiredFieldThrows() {
        let json = """
            { "hidden_size": 64, "num_hidden_layers": 2 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIBailingHybridConfiguration.self, from: Data(json.utf8)))
    }

    func testValidationRejectsHeadMismatch() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 3,
                "intermediate_size": 128,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNegativeNextnLayers() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "num_nextn_predict_layers": -1
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIBailingHybridConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
