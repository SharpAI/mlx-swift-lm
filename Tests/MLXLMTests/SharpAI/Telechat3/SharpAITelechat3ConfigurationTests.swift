// SharpAI fork overlay tests for the telechat3 config. Not part of upstream.
// Offline only: inline JSON, no checkpoint.
// UNVERIFIED: key names and defaults follow the Llama layout. Written without compiling.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAITelechat3ConfigurationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["TeleChat3ForCausalLM"],
          "model_type": "telechat3",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "vocab_size": 1000,
          "rms_norm_eps": 0.00001,
          "rope_theta": 10000.0,
          "max_position_embeddings": 4096,
          "tie_word_embeddings": false,
          "rope_scaling": {
            "type": "telechat3-yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 1024
          }
        }
        """

    private func decode(_ json: String) throws -> SharpAITelechat3Configuration {
        try JSONDecoder.json5().decode(SharpAITelechat3Configuration.self, from: Data(json.utf8))
    }

    func testDecodesCoreKeysFromInlineJSON() throws {
        let configuration = try decode(validJSON)

        XCTAssertEqual(configuration.modelType, "telechat3")
        XCTAssertEqual(configuration.architectures, ["TeleChat3ForCausalLM"])
        XCTAssertEqual(configuration.hiddenSize, 64)
        XCTAssertEqual(configuration.hiddenLayers, 2)
        XCTAssertEqual(configuration.intermediateSize, 128)
        XCTAssertEqual(configuration.attentionHeads, 8)
        XCTAssertEqual(configuration.kvHeads, 2)
        XCTAssertEqual(configuration.vocabularySize, 1000)
        XCTAssertEqual(configuration.ropeTheta, 10_000)
        XCTAssertEqual(configuration.maxPositionEmbeddings, 4096)
        XCTAssertFalse(configuration.tieWordEmbeddings)
        XCTAssertEqual(configuration.resolvedHeadDim, 16)
    }

    func testKeepsTelechat3YarnRopeScaling() throws {
        let configuration = try decode(validJSON)

        XCTAssertEqual(
            configuration.ropeScaling?["type"], StringOrNumber.string("telechat3-yarn"))
        XCTAssertEqual(configuration.ropeScaling?["factor"]?.asFloat(), 4.0)
    }

    func testOptionalKeysUseDefaults() throws {
        let json = """
            {
              "model_type": "telechat3",
              "hidden_size": 64,
              "intermediate_size": 128,
              "num_hidden_layers": 2,
              "num_attention_heads": 8,
              "vocab_size": 1000
            }
            """
        let configuration = try decode(json)

        XCTAssertEqual(configuration.kvHeads, 8)
        XCTAssertNil(configuration.headDim)
        XCTAssertEqual(configuration.resolvedHeadDim, 8)
        XCTAssertEqual(configuration.ropeTheta, 10_000)
        XCTAssertFalse(configuration.ropeTraditional)
        XCTAssertNil(configuration.ropeScaling)
        XCTAssertFalse(configuration.attentionBias)
        XCTAssertFalse(configuration.mlpBias)
        XCTAssertFalse(configuration.tieWordEmbeddings)
    }

    func testLlamaConfigurationCarriesFields() throws {
        let llama = try decode(validJSON).llamaConfiguration

        XCTAssertEqual(llama.hiddenSize, 64)
        XCTAssertEqual(llama.attentionHeads, 8)
        XCTAssertEqual(llama.kvHeads, 2)
        XCTAssertEqual(llama.resolvedHeadDimensions, 16)
        XCTAssertEqual(llama.vocabularySize, 1000)
        XCTAssertEqual(llama.modelType, "telechat3")
    }

    func testValidConfigPassesValidation() throws {
        try decode(validJSON).validateModelConfiguration()
    }

    func testWrongArchitectureFailsValidation() throws {
        let json = validJSON.replacingOccurrences(
            of: "TeleChat3ForCausalLM", with: "LlamaForCausalLM")

        XCTAssertThrowsError(try decode(json).validateModelConfiguration())
    }

    func testHeadCountMustDivideByKVHeads() throws {
        let json = validJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")

        XCTAssertThrowsError(try decode(json).validateModelConfiguration())
    }

    func testUnsupportedRopeTypeFailsValidation() throws {
        let json = validJSON.replacingOccurrences(
            of: "telechat3-yarn", with: "not-a-rope-type")

        XCTAssertThrowsError(try decode(json).validateModelConfiguration())
    }
}
