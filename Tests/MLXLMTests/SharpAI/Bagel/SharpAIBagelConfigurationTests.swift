// SharpAI fork overlay tests for bagel config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIBagelConfigurationTests: XCTestCase {

    func testDecodesTextAndVisionConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self,
            from: Data(SharpAIBagelTestFixtures.tinyJSON.utf8))

        XCTAssertEqual(config.modelType, "bagel")
        XCTAssertEqual(config.architectures, ["BagelForConditionalGeneration"])
        XCTAssertEqual(config.connectorAct, "gelu_pytorch_tanh")
        XCTAssertEqual(config.vitMaxNumPatchPerSide, 2)
        XCTAssertEqual(config.imageTokenIndex, 31)

        let text = config.textConfiguration
        XCTAssertEqual(text.hiddenSize, 16)
        XCTAssertEqual(text.hiddenLayers, 2)
        XCTAssertEqual(text.intermediateSize, 32)
        XCTAssertEqual(text.attentionHeads, 4)
        XCTAssertEqual(text.kvHeads, 2)
        XCTAssertEqual(text.vocabularySize, 32)
        XCTAssertEqual(text.ropeTheta, 10000)
        XCTAssertFalse(text.tieWordEmbeddings)
        XCTAssertTrue(text.qkNorm)

        let vision = config.visionConfiguration
        XCTAssertEqual(vision.hiddenSize, 8)
        XCTAssertEqual(vision.hiddenLayers, 1)
        XCTAssertEqual(vision.attentionHeads, 2)
        XCTAssertEqual(vision.channels, 3)
        XCTAssertEqual(vision.imageSize, 28)
        XCTAssertEqual(vision.patchSize, 14)

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testOptionalKeysUseDefaults() throws {
        let json = """
            {
                "llm_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "intermediate_size": 32,
                    "num_attention_heads": 4,
                    "vocab_size": 32,
                    "qk_norm": true
                },
                "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "image_size": 28,
                    "patch_size": 14
                }
            }
            """
        // UNVERIFIED: defaults are assumptions, pinned here so a change is visible.
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "bagel")
        XCTAssertNil(config.architectures)
        XCTAssertEqual(config.connectorAct, "gelu_pytorch_tanh")
        XCTAssertEqual(config.vitMaxNumPatchPerSide, 70)
        XCTAssertEqual(config.imageTokenIndex, 151655)
        XCTAssertEqual(config.textConfiguration.kvHeads, 4)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 1_000_000)
        XCTAssertEqual(config.textConfiguration.rmsNormEps, 1e-6)
        XCTAssertEqual(config.visionConfiguration.channels, 3)
        XCTAssertEqual(config.visionConfiguration.layerNormEps, 1e-6)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testValidationRequiresQKNorm() throws {
        let json = SharpAIBagelTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"qk_norm\": true", with: "\"qk_norm\": false")
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsWrongArchitecture() throws {
        let json = SharpAIBagelTestFixtures.tinyJSON.replacingOccurrences(
            of: "BagelForConditionalGeneration", with: "SomethingElse")
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsUnsupportedConnectorAct() throws {
        let json = SharpAIBagelTestFixtures.tinyJSON.replacingOccurrences(
            of: "gelu_pytorch_tanh", with: "relu")
        let config = try JSONDecoder.json5().decode(
            SharpAIBagelConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
