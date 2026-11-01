// SharpAI fork overlay tests for the param2moe configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIParam2moeConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIParam2moeConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIParam2moeConfiguration.self, from: Data(json.utf8))
    }

    func testFullFixtureDecodesCheckpointValues() throws {
        let config = try decode(SharpAIParam2moeTestFixture.fullJSON)

        XCTAssertEqual(config.modelType, "param2moe")
        XCTAssertEqual(config.vocabularySize, 128_000)
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.hiddenLayers, 21)
        XCTAssertEqual(config.intermediateSize, 9216)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.headDim, 64)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertEqual(config.partialRotaryFactor, 1.0)
        XCTAssertEqual(config.rotaryDimensions, 64)
        XCTAssertTrue(config.useQKNorm)
        XCTAssertFalse(config.useQKVBias)
        XCTAssertFalse(config.useBias)
        XCTAssertTrue(config.tieWordEmbeddings)
    }

    func testFullFixtureDecodesMoELayout() throws {
        let config = try decode(SharpAIParam2moeTestFixture.fullJSON)

        XCTAssertEqual(config.firstKDenseReplace, 1)
        XCTAssertEqual(config.numExperts, 64)
        XCTAssertEqual(config.numExpertsPerTok, 6)
        XCTAssertEqual(config.moeIntermediateSize, 2048)
        XCTAssertEqual(config.numSharedExperts, 2)
        // The shared width is the total, so it is not multiplied by num_shared_experts again.
        XCTAssertEqual(config.moeSharedExpertIntermediateSize, 4096)
        XCTAssertEqual(config.routedScalingFactor, 2.5)
        XCTAssertEqual(config.scoreFunction, "sigmoid")
        XCTAssertTrue(config.normTopkProb)
        XCTAssertEqual(config.nGroup, 1)
        XCTAssertEqual(config.topkGroup, 1)
        XCTAssertFalse(config.isSparseLayer(0))
        XCTAssertTrue(config.isSparseLayer(1))
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRopeThetaFallsBackToNestedRopeParameters() throws {
        let json = try SharpAIParam2moeTestFixture.tinyJSON()
        var object = try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any] ?? [:]
        object.removeValue(forKey: "rope_theta")
        object.removeValue(forKey: "partial_rotary_factor")
        object["rope_parameters"] = ["rope_theta": 250_000, "partial_rotary_factor": 0.5]
        let data = try JSONSerialization.data(withJSONObject: object)

        let config = try JSONDecoder.json5().decode(SharpAIParam2moeConfiguration.self, from: data)

        XCTAssertEqual(config.ropeTheta, 250_000)
        XCTAssertEqual(config.partialRotaryFactor, 0.5)
        XCTAssertEqual(config.rotaryDimensions, 4)
    }

    func testDefaultsApplyWhenOptionalKeysAreAbsent() throws {
        let minimal = """
            {
              "vocab_size": 100,
              "hidden_size": 32,
              "num_hidden_layers": 2,
              "intermediate_size": 64,
              "num_attention_heads": 4,
              "num_key_value_heads": 2
            }
            """
        let config = try decode(minimal)

        XCTAssertEqual(config.headDim, 8)
        XCTAssertEqual(config.ropeTheta, 10_000)
        XCTAssertEqual(config.numExperts, 0)
        XCTAssertFalse(config.isSparseLayer(1))
        XCTAssertEqual(config.scoreFunction, "sigmoid")
        XCTAssertEqual(config.nGroup, 1)
        XCTAssertEqual(config.topkGroup, 1)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsTopKLargerThanExpertCount() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(["num_experts_per_tok": 9]))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIParam2moeError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsGroupsThatDoNotDivideExperts() throws {
        let config = try decode(SharpAIParam2moeTestFixture.tinyJSON(["n_group": 3]))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsGroupsOfOneExpert() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(
                ["n_group": 4, "topk_group": 1, "num_experts_per_tok": 1]))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIParam2moeError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsTopKBeyondKeptGroups() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(
                ["n_group": 2, "topk_group": 1, "num_experts_per_tok": 3]))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsHeadsThatDoNotDivideKVHeads() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(["num_key_value_heads": 3]))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsUnknownScoreFunction() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(["score_function": "relu"]))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIParam2moeError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsForeignArchitecture() throws {
        let config = try decode(
            SharpAIParam2moeTestFixture.tinyJSON(["architectures": ["LlamaForCausalLM"]]))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
