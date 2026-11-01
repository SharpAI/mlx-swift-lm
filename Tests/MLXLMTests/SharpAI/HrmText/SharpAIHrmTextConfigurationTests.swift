// SharpAI fork overlay tests for hrm_text config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIHrmTextConfigurationTests: XCTestCase {

    // Key names follow the HRM-Text-1B fixture. Sizes here are test values, not the fixture's.
    private let fullJSON = """
        {
            "model_type": "hrm_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 600000.0,
            "tie_word_embeddings": true,
            "H_cycles": 2,
            "L_cycles": 3,
            "L_bp_cycles": [0, 2]
        }
        """

    func testDecodesAllFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(fullJSON.utf8))

        XCTAssertEqual(config.modelType, "hrm_text")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.rmsNormEps, 1e-5, accuracy: 1e-9)
        XCTAssertEqual(config.ropeTheta, 600_000.0)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.hCycles, 2)
        XCTAssertEqual(config.lCycles, 3)
        XCTAssertEqual(config.lBpCycles, [0, 2])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        // Defaults match the HRM-Text-1B fixture.
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "H_cycles": 2,
                "L_cycles": 3,
                "L_bp_cycles": [0]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "hrm_text")
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000.0)
        XCTAssertFalse(config.tieWordEmbeddings)
    }

    func testMissingRequiredFieldThrows() {
        let json = """
            { "hidden_size": 64, "num_hidden_layers": 2 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIHrmTextConfiguration.self, from: Data(json.utf8)))
    }

    func testMissingLoopFieldThrows() {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "L_cycles": 3,
                "L_bp_cycles": [0]
            }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIHrmTextConfiguration.self, from: Data(json.utf8)))
    }

    func testValidationRejectsHeadMismatch() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 3,
                "intermediate_size": 128,
                "vocab_size": 512,
                "H_cycles": 2,
                "L_cycles": 3,
                "L_bp_cycles": [0]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsNonPositiveSize() throws {
        let json = """
            {
                "hidden_size": 0,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "intermediate_size": 128,
                "vocab_size": 512,
                "H_cycles": 2,
                "L_cycles": 3,
                "L_bp_cycles": [0]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIHrmTextConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
