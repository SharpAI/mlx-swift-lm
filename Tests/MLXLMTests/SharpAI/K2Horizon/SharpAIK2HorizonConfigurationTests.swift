// SharpAI fork overlay tests for k2_horizon config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

/// Shared inline fixture. Tiny shapes, no weights.
enum SharpAIK2HorizonTestFixtures {
    static func json(extra: String = "") -> String {
        """
            {
                "model_type": "k2_horizon",
                "architectures": ["K2HorizonForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 128\(extra)
            }
            """
    }

    static func configuration(extra: String = "") throws -> SharpAIK2HorizonConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIK2HorizonConfiguration.self, from: Data(json(extra: extra).utf8))
    }
}

final class SharpAIK2HorizonConfigurationTests: XCTestCase {

    func testDecodesBaseFieldsAndDefaults() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration()

        XCTAssertEqual(config.modelType, "k2_horizon")
        XCTAssertEqual(config.vocabularySize, 32)
        XCTAssertEqual(config.hiddenSize, 8)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.intermediateSize, 16)
        XCTAssertEqual(config.attentionHeads, 2)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertNil(config.headDimensions)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertNil(config.ropeScaling)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.mlpBias)
        XCTAssertEqual(config.rmsNormGroups, 1)
    }

    func testKvHeadsDefaultsToAttentionHeads() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration()
        XCTAssertEqual(config.kvHeads, config.attentionHeads)
    }

    func testRopeParametersSetsThetaAndScaling() throws {
        // UNVERIFIED: rope_parameters key layout is assumed (transformers v5 style).
        let config = try SharpAIK2HorizonTestFixtures.configuration(
            extra: """
                ,
                    "rope_parameters": {
                        "rope_type": "llama3",
                        "rope_theta": 500000.0,
                        "factor": 8.0,
                        "low_freq_factor": 1.0,
                        "high_freq_factor": 4.0,
                        "original_max_position_embeddings": 8192
                    }
                """)

        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertEqual(config.ropeScaling?["factor"]?.asFloat(), 8)
        XCTAssertEqual(config.ropeScaling?["rope_type"], StringOrNumber.string("llama3"))
        XCTAssertNil(config.ropeScaling?["rope_theta"])
    }

    func testRopeParametersDefaultTypeClearsScaling() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration(
            extra: """
                ,
                    "rope_theta": 1000.0,
                    "rope_parameters": { "rope_type": "default", "rope_theta": 20000.0 }
                """)

        XCTAssertEqual(config.ropeTheta, 20_000)
        XCTAssertNil(config.ropeScaling)
    }

    func testTopLevelRopeFieldsUsedWithoutRopeParameters() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration(
            extra: """
                ,
                    "rope_theta": 1000.0,
                    "rope_scaling": { "type": "linear", "factor": 2.0 }
                """)

        XCTAssertEqual(config.ropeTheta, 1_000)
        XCTAssertEqual(config.ropeScaling?["type"], StringOrNumber.string("linear"))
    }

    func testRmsNormGroupsDecodes() throws {
        // UNVERIFIED: rms_norm_groups key name is assumed.
        let config = try SharpAIK2HorizonTestFixtures.configuration(
            extra: ", \"rms_norm_groups\": 2")
        XCTAssertEqual(config.rmsNormGroups, 2)
    }

    func testRmsNormGroupsMustDivideHiddenSize() {
        XCTAssertThrowsError(
            try SharpAIK2HorizonTestFixtures.configuration(
                extra: ", \"rms_norm_groups\": 3"))
    }

    func testLlamaViewCarriesResolvedRope() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration(
            extra: """
                ,
                    "rope_parameters": {
                        "rope_type": "linear", "rope_theta": 5000.0, "factor": 4.0
                    }
                """)

        let llama = config.llama
        XCTAssertEqual(llama.ropeTheta, 5_000)
        XCTAssertEqual(llama.ropeScaling?["factor"]?.asFloat(), 4)
        XCTAssertEqual(llama.resolvedHeadDimensions, 4)
    }

    func testValidationRejectsZeroAttentionHeads() throws {
        let json = SharpAIK2HorizonTestFixtures.json().replacingOccurrences(
            of: "\"num_attention_heads\": 2", with: "\"num_attention_heads\": 0")
        let config = try JSONDecoder.json5().decode(
            SharpAIK2HorizonConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsIndivisibleHiddenSizeWithoutHeadDim() throws {
        let json = SharpAIK2HorizonTestFixtures.json().replacingOccurrences(
            of: "\"num_attention_heads\": 2", with: "\"num_attention_heads\": 3"
        ).replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIK2HorizonConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsKvHeadsNotDividingHeads() throws {
        let json = SharpAIK2HorizonTestFixtures.json().replacingOccurrences(
            of: "\"num_key_value_heads\": 2", with: "\"num_key_value_heads\": 3")
        let config = try JSONDecoder.json5().decode(
            SharpAIK2HorizonConfiguration.self, from: Data(json.utf8))
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationAcceptsBaseFixture() throws {
        let config = try SharpAIK2HorizonTestFixtures.configuration()
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
