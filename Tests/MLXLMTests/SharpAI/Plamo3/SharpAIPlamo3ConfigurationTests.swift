// SharpAI fork overlay tests for the plamo3 config. Not part of upstream.
// Offline only: inline JSON, no checkpoint.
// UNVERIFIED: the key names are guesses (gated repo, no config.json here). Written without compiling.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIPlamo3ConfigurationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["Plamo3ForCausalLM"],
          "model_type": "plamo3",
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
          "sliding_window": 512,
          "tie_word_embeddings": false
        }
        """

    private func decode(_ json: String) throws -> SharpAIPlamo3Configuration {
        try JSONDecoder.json5().decode(SharpAIPlamo3Configuration.self, from: Data(json.utf8))
    }

    func testDecodesCoreKeysFromInlineJSON() throws {
        let configuration = try decode(validJSON)

        XCTAssertEqual(configuration.modelType, "plamo3")
        XCTAssertEqual(configuration.architectures, ["Plamo3ForCausalLM"])
        XCTAssertEqual(configuration.hiddenSize, 64)
        XCTAssertEqual(configuration.intermediateSize, 128)
        XCTAssertEqual(configuration.hiddenLayers, 2)
        XCTAssertEqual(configuration.attentionHeads, 8)
        XCTAssertEqual(configuration.kvHeads, 2)
        XCTAssertEqual(configuration.vocabularySize, 1000)
        XCTAssertEqual(configuration.ropeTheta, 10_000)
        XCTAssertEqual(configuration.maxPositionEmbeddings, 4096)
        XCTAssertEqual(configuration.slidingWindow, 512)
        XCTAssertFalse(configuration.tieWordEmbeddings)
        XCTAssertEqual(configuration.resolvedHeadDim, 16)
    }

    func testOptionalKeysUseDefaults() throws {
        let json = """
            {
              "model_type": "plamo3",
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 8,
              "vocab_size": 1000
            }
            """
        let configuration = try decode(json)

        XCTAssertEqual(configuration.architectures, [])
        XCTAssertEqual(configuration.kvHeads, 8)
        XCTAssertNil(configuration.headDim)
        XCTAssertEqual(configuration.resolvedHeadDim, 8)
        XCTAssertNil(configuration.intermediateSize)
        XCTAssertNil(configuration.slidingWindow)
        XCTAssertFalse(configuration.tieWordEmbeddings)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingRequiredKeyFailsToDecode() {
        let json = """
            {
              "model_type": "plamo3",
              "hidden_size": 64,
              "num_attention_heads": 8,
              "vocab_size": 1000
            }
            """

        XCTAssertThrowsError(try decode(json))
    }

    func testValidateAcceptsExpectedConfiguration() throws {
        let configuration = try decode(validJSON)

        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testValidateRejectsWrongModelType() throws {
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"plamo3\"", with: "\"model_type\": \"plamo2\"")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIPlamo3Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidateRejectsWrongArchitecture() throws {
        let json = validJSON.replacingOccurrences(
            of: "Plamo3ForCausalLM", with: "LlamaForCausalLM")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIPlamo3Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidateRejectsHeadsNotDivisibleByKVHeads() throws {
        let json = validJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIPlamo3Error,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
