// SharpAI fork overlay tests for the minimax_m3_text config. Not part of upstream.
// Offline only: inline JSON, no checkpoint.
// UNVERIFIED: the key names come from the model spec and the minimax config. Written without compiling.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMinimaxM3TextConfigurationTests: XCTestCase {

    private let validJSON = """
        {
          "architectures": ["MiniMaxM3SparseForCausalLM"],
          "model_type": "minimax_m3_text",
          "hidden_size": 64,
          "intermediate_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "vocab_size": 1000,
          "rms_norm_eps": 0.00001,
          "rope_theta": 5000000.0,
          "rotary_dim": 8,
          "max_position_embeddings": 4096,
          "num_local_experts": 4,
          "num_experts_per_tok": 2,
          "tie_word_embeddings": true
        }
        """

    private func decode(_ json: String) throws -> SharpAIMinimaxM3TextConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIMinimaxM3TextConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesCoreKeysFromInlineJSON() throws {
        let configuration = try decode(validJSON)

        XCTAssertEqual(configuration.modelType, "minimax_m3_text")
        XCTAssertEqual(configuration.architectures, ["MiniMaxM3SparseForCausalLM"])
        XCTAssertEqual(configuration.hiddenSize, 64)
        XCTAssertEqual(configuration.intermediateSize, 128)
        XCTAssertEqual(configuration.hiddenLayers, 2)
        XCTAssertEqual(configuration.attentionHeads, 8)
        XCTAssertEqual(configuration.kvHeads, 2)
        XCTAssertEqual(configuration.vocabularySize, 1000)
        XCTAssertEqual(configuration.ropeTheta, 5_000_000)
        XCTAssertEqual(configuration.rotaryDim, 8)
        XCTAssertEqual(configuration.numLocalExperts, 4)
        XCTAssertEqual(configuration.numExpertsPerTok, 2)
        XCTAssertTrue(configuration.tieWordEmbeddings)
        XCTAssertEqual(configuration.resolvedHeadDim, 16)
    }

    func testOptionalKeysUseDefaults() throws {
        let json = """
            {
              "model_type": "minimax_m3_text",
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
        XCTAssertFalse(configuration.tieWordEmbeddings)
        XCTAssertNil(configuration.numLocalExperts)
        XCTAssertNoThrow(try configuration.validateModelConfiguration())
    }

    func testMissingRequiredKeyFailsToDecode() {
        let json = """
            {
              "model_type": "minimax_m3_text",
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
            of: "\"model_type\": \"minimax_m3_text\"", with: "\"model_type\": \"minimax\"")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3TextError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testValidateRejectsWrongArchitecture() throws {
        let json = validJSON.replacingOccurrences(
            of: "MiniMaxM3SparseForCausalLM", with: "LlamaForCausalLM")
        let configuration = try decode(json)

        XCTAssertThrowsError(try configuration.validateModelConfiguration()) { error in
            guard let typed = error as? SharpAIMinimaxM3TextError,
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
            guard let typed = error as? SharpAIMinimaxM3TextError,
                case .invalidConfiguration = typed
            else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }
}
