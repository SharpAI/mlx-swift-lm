// SharpAI fork overlay tests for hyperclovax config decoding. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIHyperclovaxConfigurationTests: XCTestCase {

    /// Values copied from the HyperCLOVAX-SEED-Think-14B fixture.
    private let fixtureJSON = """
        {
            "architectures": ["HyperCLOVAXForCausalLM"],
            "attention_bias": false,
            "attention_multiplier": 0.0078125,
            "embedding_multiplier": 10.0,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 6144,
            "intermediate_size": 14336,
            "logits_scaling": 0.125,
            "max_position_embeddings": 131072,
            "mlp_bias": false,
            "model_type": "hyperclovax",
            "num_attention_heads": 48,
            "num_hidden_layers": 38,
            "num_key_value_heads": 8,
            "residual_multiplier": 1.0,
            "rms_norm_eps": 1e-05,
            "rope_scaling": null,
            "rope_theta": 100000000,
            "tie_word_embeddings": false,
            "use_post_norm": true,
            "vocab_size": 110592
        }
        """

    private let minimalJSON = """
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "vocab_size": 512
        }
        """

    func testDecodesFixtureFields() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxConfiguration.self, from: Data(fixtureJSON.utf8))

        XCTAssertEqual(config.modelType, "hyperclovax")
        XCTAssertEqual(config.hiddenSize, 6144)
        XCTAssertEqual(config.hiddenLayers, 38)
        XCTAssertEqual(config.intermediateSize, 14336)
        XCTAssertEqual(config.attentionHeads, 48)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.vocabularySize, 110_592)
        XCTAssertEqual(config.maxPositionEmbeddings, 131_072)
        XCTAssertEqual(config.ropeTheta, 100_000_000)
        XCTAssertNil(config.ropeScaling)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertEqual(config.attentionMultiplier, 0.0078125)
        XCTAssertEqual(config.embeddingMultiplier, 10.0)
        XCTAssertEqual(config.residualMultiplier, 1.0)
        XCTAssertEqual(config.logitsScaling, 0.125)
        XCTAssertTrue(config.usePostNorm)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalFieldsUseDefaults() throws {
        // UNVERIFIED: the fallback values are guesses, pinned here so a change is visible.
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxConfiguration.self, from: Data(minimalJSON.utf8))

        XCTAssertEqual(config.modelType, "hyperclovax")
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertNil(config.maxPositionEmbeddings)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.attentionMultiplier, pow(Float(8), -0.5), accuracy: 1e-6)
        XCTAssertEqual(config.embeddingMultiplier, 1.0)
        XCTAssertEqual(config.residualMultiplier, 1.0)
        XCTAssertEqual(config.logitsScaling, 1.0)
        XCTAssertFalse(config.usePostNorm)
    }

    func testMissingRequiredFieldThrows() {
        let json = """
            { "hidden_size": 64, "num_hidden_layers": 2 }
            """
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIHyperclovaxConfiguration.self, from: Data(json.utf8)))
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
            SharpAIHyperclovaxConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsZeroLogitsScaling() throws {
        let json = """
            {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "vocab_size": 512,
                "logits_scaling": 0.0
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIHyperclovaxConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
