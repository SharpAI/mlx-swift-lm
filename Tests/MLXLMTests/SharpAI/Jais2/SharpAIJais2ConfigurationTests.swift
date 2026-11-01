// SharpAI fork overlay tests for jais2 config decoding. Not part of upstream.
// UNVERIFIED: fixture values are synthetic, not from a real Jais 2 checkpoint.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIJais2ConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIJais2Configuration {
        try JSONDecoder.json5().decode(
            SharpAIJais2Configuration.self, from: Data(json.utf8))
    }

    func testDecodesExplicitFields() throws {
        let config = try decode(
            """
            {
                "model_type": "jais2",
                "architectures": ["Jais2ForCausalLM"],
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 3,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "layer_norm_eps": 1e-6,
                "rope_theta": 500000.0,
                "max_position_embeddings": 128,
                "tie_word_embeddings": true,
                "hidden_act": "relu2"
            }
            """)

        XCTAssertEqual(config.modelType, "jais2")
        XCTAssertEqual(config.vocabularySize, 64)
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.intermediateSize, 32)
        XCTAssertEqual(config.hiddenLayers, 3)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.layerNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 500_000, accuracy: 1)
        XCTAssertEqual(config.maxPositionEmbeddings, 128)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.hiddenAct, "relu2")
    }

    func testAppliesDefaultsWhenOptionalFieldsAbsent() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
            """)

        XCTAssertEqual(config.modelType, "jais2")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.layerNormEps, 1e-5, accuracy: 1e-12)
        XCTAssertEqual(config.ropeTheta, 10_000, accuracy: 1e-3)
        XCTAssertNil(config.maxPositionEmbeddings)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertTrue(config.attentionBias)
        XCTAssertTrue(config.mlpBias)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertNil(config.hiddenAct)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testReadsRopeThetaFromRopeParametersAndExplicitHeadDim() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "head_dim": 8,
                "attention_bias": false,
                "mlp_bias": false,
                "rope_parameters": { "rope_type": "default", "rope_theta": 250000.0 }
            }
            """)

        XCTAssertEqual(config.ropeTheta, 250_000, accuracy: 1)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingRequiredFieldThrows() {
        XCTAssertThrowsError(
            try decode(
                """
                {
                    "vocab_size": 64,
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4
                }
                """))
    }

    func testRejectsHiddenSizeNotDivisibleByHeads() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 18,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertTrue(error is SharpAIJais2Error)
        }
    }

    func testRejectsHeadsNotDivisibleByKVHeads() throws {
        let config = try decode(
            """
            {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 3
            }
            """)

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertTrue(error is SharpAIJais2Error)
        }
    }
}
