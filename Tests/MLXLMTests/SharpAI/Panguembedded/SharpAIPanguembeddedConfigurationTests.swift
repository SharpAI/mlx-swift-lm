// SharpAI fork overlay tests for the PanguEmbedded configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPanguembeddedConfigurationTests: XCTestCase {

    /// Values from scripts/conformance/offline/fixtures/PanguEmbedded (openPangu-Embedded-7B-V1.1).
    private let fixtureJSON = """
        {
            "architectures": ["PanguEmbeddedForCausalLM"],
            "bias": true,
            "bos_token_id": 1,
            "eos_token_id": 45892,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 12800,
            "max_position_embeddings": 32768,
            "model_type": "PanguEmbedded",
            "num_attention_heads": 32,
            "num_hidden_layers": 34,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 16000000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 153376
        }
        """

    private func decode(_ json: String) throws -> SharpAIPanguembeddedConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIPanguembeddedConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesFixtureShape() throws {
        let config = try decode(fixtureJSON).llamaConfiguration

        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.hiddenLayers, 34)
        XCTAssertEqual(config.intermediateSize, 12800)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.vocabularySize, 153376)
        XCTAssertEqual(config.resolvedHeadDimensions, 128)
        XCTAssertEqual(config.ropeTheta, 16_000_000)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.modelType, "PanguEmbedded")
    }

    func testBiasKeyEnablesAttentionBias() throws {
        let config = try decode(fixtureJSON).llamaConfiguration
        XCTAssertTrue(config.attentionBias)
    }

    func testMLPBiasIsNotTakenFromBiasKey() throws {
        let config = try decode(fixtureJSON).llamaConfiguration
        XCTAssertFalse(config.mlpBias)
    }

    func testAttentionBiasKeyWinsOverBiasKey() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"bias\": true,", with: "\"bias\": true, \"attention_bias\": false,")
        let config = try decode(json).llamaConfiguration
        XCTAssertFalse(config.attentionBias)
    }

    func testMissingBiasKeysDefaultToNoAttentionBias() throws {
        let json = fixtureJSON.replacingOccurrences(of: "\"bias\": true,", with: "")
        let config = try decode(json).llamaConfiguration
        XCTAssertFalse(config.attentionBias)
    }

    func testValidationRejectsHeadsNotDivisibleByKVHeads() throws {
        let json = fixtureJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 8,", with: "\"num_key_value_heads\": 5,")
        let config = try decode(json)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationAcceptsFixtureShape() throws {
        let config = try decode(fixtureJSON)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
