// SharpAI fork overlay tests for the iquestcoder configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIIquestcoderConfigurationTests: XCTestCase {

    // Shape values are illustrative test values, not the published IQuest-Coder checkpoint.
    private let inlineJSON = """
        {
            "model_type": "iquestcoder",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 1e-05,
            "vocab_size": 512,
            "rope_theta": 500000.0,
            "tie_word_embeddings": false
        }
        """

    func testDecodesLlamaFieldsFromInlineJSON() throws {
        let config = try JSONDecoder.json5().decode(
            LlamaConfiguration.self, from: Data(inlineJSON.utf8))

        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.resolvedHeadDimensions, 16)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.modelType, "iquestcoder")
    }

    func testHeadDimDefaultsToHiddenOverHeads() throws {
        let json = """
            {
                "model_type": "iquestcoder",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "rms_norm_eps": 1e-05,
                "vocab_size": 512
            }
            """
        let config = try JSONDecoder.json5().decode(
            LlamaConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.resolvedHeadDimensions, 8)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.ropeTheta, 10_000)
    }

    func testMissingRequiredFieldFailsDecoding() {
        let json = """
            { "model_type": "iquestcoder", "hidden_size": 64 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(LlamaConfiguration.self, from: Data(json.utf8)))
    }

    func testToolCallFormatIsNotInferredFromLlama3Vocabulary() throws {
        // Llama 3 tool-call inference is gated on model_type == "llama". This alias must not
        // get .llama3 from its vocabulary size. UNVERIFIED: the real iquestcoder chat template
        // tool format is not in the repo, so nil is the narrow answer here.
        let json = """
            {
                "model_type": "iquestcoder",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "rms_norm_eps": 1e-05,
                "vocab_size": 128256
            }
            """
        let config = try JSONDecoder.json5().decode(
            LlamaConfiguration.self, from: Data(json.utf8))

        XCTAssertNil(LlamaModel(config).toolCallFormat)
    }
}
