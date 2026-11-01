// SharpAI fork overlay tests for the interns2_mobius configuration. Not part of upstream.

import Foundation
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIInterns2MobiusConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "interns2_mobius",
            "architectures": ["InternS2MobiusForConditionalGeneration"],
            "mtp_num_experts": 8
        }
        """

    func testDecodesModelTypeArchitecturesAndMTPExperts() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "interns2_mobius")
        XCTAssertEqual(config.architectures, ["InternS2MobiusForConditionalGeneration"])
        XCTAssertEqual(config.mtpNumExperts, 8)
    }

    /// Real layout: the text model sits under text_config. Values are from the HF config.json.
    func testDecodesNestedTextConfig() throws {
        let json = """
            {
                "model_type": "interns2_mobius",
                "text_config": {
                    "model_type": "interns2_mobius_text",
                    "num_hidden_layers": 40,
                    "hidden_size": 2048,
                    "num_experts": 2560,
                    "num_experts_per_tok": 8,
                    "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512,
                    "num_blocks": 4,
                    "head_dim": 256,
                    "partial_rotary_factor": 0.25,
                    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))

        let text = config.textConfiguration
        XCTAssertEqual(text.hiddenLayers, 40)
        XCTAssertEqual(text.hiddenSize, 2048)
        XCTAssertEqual(text.numExperts, 2560)
        XCTAssertEqual(text.numExpertsPerTok, 8)
        XCTAssertEqual(text.numBlocks, 4)
        XCTAssertEqual(text.rotaryDim, 64)
        XCTAssertEqual(text.ropeTheta, 10_000_000)
        XCTAssertEqual(text.layerTypes.count, 40)
        XCTAssertEqual(text.layerTypes[3], "full_attention")
        XCTAssertEqual(text.layerTypes[0], "linear_attention")
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testDecodesMTPArchitectureName() throws {
        let json = """
            {
                "model_type": "interns2_mobius",
                "architectures": ["InternS2MobiusMTP"]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.architectures, ["InternS2MobiusMTP"])
    }

    func testMissingMTPExpertsDefaultsToZero() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data("{}".utf8))

        XCTAssertEqual(config.modelType, "interns2_mobius")
        XCTAssertEqual(config.architectures, [])
        XCTAssertEqual(config.mtpNumExperts, 0)
    }

    func testValidationAcceptsInterns2MobiusModelType() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRejectsOtherModelType() throws {
        let json = """
            { "model_type": "llama" }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsNegativeMTPExperts() throws {
        let json = """
            { "model_type": "interns2_mobius", "mtp_num_experts": -1 }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIInterns2MobiusConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
