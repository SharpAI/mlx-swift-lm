// SharpAI fork overlay tests for the step5_mtp configuration. Not part of upstream.
// UNVERIFIED: these pin the decoder's own defaults and key names. They do not prove the real
// checkpoint layout, because no Step5 config source is on this machine.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIStep5MtpConfigurationTests: XCTestCase {

    private let flatJSON = """
        {
            "model_type": "step5_mtp",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 8,
            "num_attention_groups": 2,
            "num_hidden_layers": 2,
            "num_nextn_predict_layers": 1,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "zero_centered": true,
            "fp32_residual_connection": false,
            "layer_types_with_mtp": ["full_attention", "sliding_attention", "full_attention"]
        }
        """

    private let nestedJSON = """
        {
            "model_type": "step5_vl",
            "architectures": ["Step5ForConditionalGeneration"],
            "vision_config": {"width": 32},
            "text_config": {
                "model_type": "step5_mtp",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 8,
                "num_attention_groups": 2,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "vocab_size": 512
            }
        }
        """

    func testDecodesFlatDraftConfig() throws {
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(flatJSON.utf8))

        XCTAssertEqual(config.modelType, "step5_mtp")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.intermediateSize, 128)
        XCTAssertEqual(config.attentionHeads, 8)
        XCTAssertEqual(config.attentionGroups, 2)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.nextnPredictLayers, 1)
        XCTAssertEqual(config.vocabularySize, 512)
        XCTAssertEqual(config.rmsNormEps, 1e-6, accuracy: 1e-12)
        XCTAssertTrue(config.zeroCentered)
        XCTAssertFalse(config.fp32ResidualConnection)
        XCTAssertEqual(config.layerTypesIncludingMTP?.count, 3)
    }

    /// UNVERIFIED: the text_config nesting is not checked against a Step5 config.
    func testDecodesNestedTextConfig() throws {
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(nestedJSON.utf8))

        XCTAssertEqual(config.modelType, "step5_mtp")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.nextnPredictLayers, 1)
    }

    /// UNVERIFIED: these defaults are not taken from any Step5 source. Pins the decoder only.
    func testDefaultsAreUnverifiedGuesses() throws {
        let json = #"{"model_type": "step5_mtp", "num_nextn_predict_layers": 1}"#
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(json.utf8))

        XCTAssertEqual(config.hiddenSize, 5120)
        XCTAssertEqual(config.intermediateSize, 13312)
        XCTAssertEqual(config.attentionHeads, 40)
        XCTAssertEqual(config.attentionGroups, 8)
        XCTAssertEqual(config.hiddenLayers, 48)
        XCTAssertEqual(config.vocabularySize, 65536)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertNil(config.headDim)
        XCTAssertNil(config.layerTypesIncludingMTP)
    }

    func testValidConfigPassesValidation() throws {
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(flatJSON.utf8))

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testZeroNextnPredictLayersFailsValidation() throws {
        let json = #"{"model_type": "step5_mtp", "num_hidden_layers": 2, "hidden_size": 64}"#
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testShortLayerTypesFailValidation() throws {
        let json = """
            {
                "model_type": "step5_mtp",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "vocab_size": 512,
                "layer_types": ["full_attention", "sliding_attention"]
            }
            """
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testLayerTypesMustMatchDensePrefixOfMTPList() throws {
        let json = """
            {
                "model_type": "step5_mtp",
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_nextn_predict_layers": 1,
                "vocab_size": 512,
                "layer_types": ["sliding_attention", "sliding_attention"],
                "layer_types_with_mtp": ["full_attention", "sliding_attention", "full_attention"]
            }
            """
        let config = try SharpAIStep5MtpConfiguration.decode(from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
