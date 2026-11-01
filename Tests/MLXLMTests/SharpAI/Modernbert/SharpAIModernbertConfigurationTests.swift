// SharpAI fork overlay tests for the modernbert configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIModernbertConfigurationTests: XCTestCase {

    /// Keys copied from the GTE-ModernColBERT fixture config.
    private let fixtureJSON = """
        {
          "architectures": ["ModernBertModel"],
          "attention_bias": false,
          "classifier_pooling": "mean",
          "decoder_bias": true,
          "global_attn_every_n_layers": 3,
          "global_rope_theta": 160000.0,
          "hidden_activation": "gelu",
          "hidden_size": 768,
          "intermediate_size": 1152,
          "local_attention": 128,
          "local_rope_theta": 10000.0,
          "max_position_embeddings": 8192,
          "mlp_bias": false,
          "model_type": "modernbert",
          "norm_bias": false,
          "norm_eps": 1e-05,
          "num_attention_heads": 12,
          "num_hidden_layers": 22,
          "position_embedding_type": "absolute",
          "vocab_size": 50370
        }
        """

    func testDecodesFixtureFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIModernbertConfiguration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(config.hiddenSize, 768)
        XCTAssertEqual(config.hiddenLayers, 22)
        XCTAssertEqual(config.intermediateSize, 1152)
        XCTAssertEqual(config.attentionHeads, 12)
        XCTAssertEqual(config.vocabularySize, 50370)
        XCTAssertEqual(config.localAttention, 128)
        XCTAssertEqual(config.localHalfWindow, 64)
        XCTAssertEqual(config.globalAttnEveryNLayers, 3)
        XCTAssertEqual(config.globalRopeTheta, 160_000)
        XCTAssertEqual(config.localRopeTheta, 10_000)
        XCTAssertEqual(config.normEps, 1e-5, accuracy: 1e-12)
        XCTAssertFalse(config.normBias)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertEqual(config.headDimensions, 64)
        XCTAssertEqual(config.modelType, "modernbert")
    }

    func testDefaultsApplyWhenOptionalKeysAreAbsent() throws {
        let json = """
            {
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "vocab_size": 32
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIModernbertConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.localAttention, 128)
        XCTAssertEqual(config.globalAttnEveryNLayers, 3)
        XCTAssertEqual(config.globalRopeTheta, 160_000)
        XCTAssertEqual(config.localRopeTheta, 10_000)
        XCTAssertEqual(config.hiddenActivation, "gelu")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsHeadsThatDoNotDivideHidden() throws {
        let json = """
            {
                "hidden_size": 10,
                "num_hidden_layers": 2,
                "intermediate_size": 16,
                "num_attention_heads": 4,
                "vocab_size": 32
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIModernbertConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsUnsupportedActivation() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"hidden_activation\": \"gelu\"", with: "\"hidden_activation\": \"silu\"")
        let config = try JSONDecoder.json5().decode(
            SharpAIModernbertConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
