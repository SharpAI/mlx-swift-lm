// SharpAI fork overlay tests for the qwen3_5_moe_text configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIQwen35MoeTextConfigurationTests: XCTestCase {

    private func decode(_ json: String) throws -> SharpAIQwen35MoeTextConfiguration {
        try JSONDecoder.json5().decode(
            SharpAIQwen35MoeTextConfiguration.self, from: Data(json.utf8))
    }

    func testDecodesFlatMoETextConfiguration() throws {
        let config = try decode(SharpAIQwen35MoeTextTestFixture.tinyJSON())

        XCTAssertEqual(config.modelType, "qwen3_5_moe_text")
        XCTAssertEqual(config.architectures, ["Qwen3_5MoeForCausalLM"])
        XCTAssertEqual(config.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 2)
        XCTAssertEqual(config.textConfiguration.numExperts, 4)
        XCTAssertEqual(config.textConfiguration.numExpertsPerTok, 2)
        XCTAssertEqual(config.textConfiguration.moeIntermediateSize, 32)
        XCTAssertEqual(config.textConfiguration.resolvedLayerTypes.count, 2)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingArchitecturesDecodesAndValidates() throws {
        let json = """
            {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 100,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 32
            }
            """
        let config = try decode(json)

        XCTAssertTrue(config.architectures.isEmpty)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testRejectsOtherModelType() throws {
        let config = try decode(SharpAIQwen35MoeTextTestFixture.tinyJSON(modelType: "qwen3_5_moe"))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            XCTAssertEqual(
                error as? SharpAIQwen35MoeTextError,
                .invalidConfiguration("model_type must be qwen3_5_moe_text, got qwen3_5_moe"))
        }
    }

    func testRejectsWrongArchitecture() throws {
        let config = try decode(
            SharpAIQwen35MoeTextTestFixture.tinyJSON(architectures: "\"LlamaForCausalLM\""))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsDenseConfiguration() throws {
        let config = try decode(SharpAIQwen35MoeTextTestFixture.tinyJSON(numExperts: 0))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testRejectsTopKLargerThanExpertCount() throws {
        let config = try decode(
            SharpAIQwen35MoeTextTestFixture.tinyJSON(numExperts: 2, numExpertsPerTok: 3))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
