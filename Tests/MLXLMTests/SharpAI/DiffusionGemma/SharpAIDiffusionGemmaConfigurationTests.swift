// SharpAI fork overlay tests for diffusion_gemma config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIDiffusionGemmaConfigurationTests: XCTestCase {

    private let nestedJSON = """
        {
            "model_type": "diffusion_gemma",
            "text_config": {
                "vocab_size": 1024,
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 4
            },
            "vision_config": {
                "model_type": "vision"
            }
        }
        """

    private let flatJSON = """
        {
            "model_type": "diffusion_gemma",
            "vocab_size": 512,
            "hidden_size": 128,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_key_value_heads": 2
        }
        """

    private func configJSON(hidden: Int, heads: Int, kvHeads: Int) -> Data {
        Data(
            """
            {
                "model_type": "diffusion_gemma",
                "text_config": {
                    "vocab_size": 1024,
                    "hidden_size": \(hidden),
                    "num_hidden_layers": 2,
                    "num_attention_heads": \(heads),
                    "num_key_value_heads": \(kvHeads)
                }
            }
            """.utf8)
    }

    func testDecodesNestedTextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDiffusionGemmaConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "diffusion_gemma")
        XCTAssertEqual(config.textDimensions.vocabSize, 1024)
        XCTAssertEqual(config.textDimensions.hiddenSize, 256)
        XCTAssertEqual(config.textDimensions.numHiddenLayers, 2)
        XCTAssertEqual(config.textDimensions.numAttentionHeads, 8)
        XCTAssertEqual(config.textDimensions.numKeyValueHeads, 4)
        XCTAssertTrue(config.hasVisionConfig)
    }

    func testFlatConfigFallsBackToTopLevelText() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDiffusionGemmaConfiguration.self, from: Data(flatJSON.utf8))

        XCTAssertEqual(config.textDimensions.vocabSize, 512)
        XCTAssertEqual(config.textDimensions.hiddenSize, 128)
        XCTAssertEqual(config.textDimensions.numHiddenLayers, 3)
        XCTAssertFalse(config.hasVisionConfig)
    }

    func testMissingRequiredTextFieldThrows() {
        let missingHidden = Data(
            """
            {
                "model_type": "diffusion_gemma",
                "text_config": {
                    "vocab_size": 1024,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4
                }
            }
            """.utf8)

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAIDiffusionGemmaConfiguration.self, from: missingHidden))
    }

    func testValidConfigPassesValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDiffusionGemmaConfiguration.self, from: Data(nestedJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testZeroHiddenSizeFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDiffusionGemmaConfiguration.self,
            from: configJSON(hidden: 0, heads: 8, kvHeads: 4))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testHeadsNotDivisibleByKvHeadsFailsValidation() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAIDiffusionGemmaConfiguration.self,
            from: configJSON(hidden: 256, heads: 8, kvHeads: 3))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
