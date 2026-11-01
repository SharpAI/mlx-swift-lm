// SharpAI fork overlay tests for iquestloopcoder configuration. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIIquestloopcoderConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "iquestloopcoder",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 512,
            "tie_word_embeddings": true
        }
        """

    func testDecodesBaseFieldsFromInlineJSON() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIIquestloopcoderConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "iquestloopcoder")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertTrue(config.tieWordEmbeddings)
    }

    func testDefaultsWhenOptionalFieldsAreMissing() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "intermediate_size": 128,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIIquestloopcoderConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "iquestloopcoder")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.ropeTheta, 500_000.0)
        XCTAssertEqual(config.loopNum, 2)
        XCTAssertEqual(config.loopWindowSize, 64)
        XCTAssertEqual(config.maxPositionEmbeddings, 8192)
        XCTAssertEqual(config.headDim, 16)  // hidden 64 / heads 4
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIIquestloopcoderConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadsNotDivisibleByKVHeadsFailsValidation() throws {
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
            SharpAIIquestloopcoderConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
