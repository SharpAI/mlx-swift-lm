// SharpAI fork overlay tests for the llama_bidirec configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILlamaBidirecConfigurationTests: XCTestCase {

    private let baseJSON = """
        {
            "model_type": "llama_bidirec",
            "architectures": ["LlamaBidirectionalModel"],
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 256
        }
        """

    func testDecodesHFKeysWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(baseJSON.utf8))

        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.vocabularySize, 128)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertFalse(config.ropeTraditional)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(config.modelType, "llama_bidirec")
        XCTAssertEqual(config.architectures, ["LlamaBidirectionalModel"])
        XCTAssertEqual(config.resolvedHeadDimensions, 4)
    }

    func testKvHeadsDefaultsToAttentionHeads() throws {
        let json = baseJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2,", with: "")
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.kvHeads, 4)
    }

    func testHeadDimensionOverrideAndRopeScalingDecode() throws {
        let json = baseJSON.replacingOccurrences(
            of: "\"max_position_embeddings\": 256",
            with: """
                "max_position_embeddings": 256,
                    "head_dim": 8,
                    "rope_theta": 500000.0,
                    "rope_scaling": {"type": "llama3", "factor": 8.0}
                """)
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.resolvedHeadDimensions, 8)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertNotNil(config.ropeScaling?["factor"])
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(baseJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadsNotMultipleOfKvHeadsFailsValidation() throws {
        let json = baseJSON.replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAILlamaBidirecConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
