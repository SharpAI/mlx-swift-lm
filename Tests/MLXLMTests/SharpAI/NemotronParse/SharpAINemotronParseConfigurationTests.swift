// SharpAI fork overlay tests for nemotron_parse config decoding. Not part of upstream.

import Foundation
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAINemotronParseConfigurationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "nemotron_parse",
            "architectures": ["NemotronParseForConditionalGeneration"],
            "encoder": {
                "model_type": "radio"
            },
            "decoder": {
                "d_model": 1024,
                "decoder_layers": 10,
                "decoder_attention_heads": 16,
                "decoder_ffn_dim": 4096,
                "vocab_size": 52000
            }
        }
        """

    func testDecodesEncoderAndDecoderBlocks() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.modelType, "nemotron_parse")
        XCTAssertEqual(config.architectures, ["NemotronParseForConditionalGeneration"])
        XCTAssertEqual(config.encoderModelType, "radio")
        XCTAssertEqual(config.decoderHiddenSize, 1024)
        XCTAssertEqual(config.decoderLayers, 10)
        XCTAssertEqual(config.decoderAttentionHeads, 16)
        XCTAssertEqual(config.decoderFFNDim, 4096)
        XCTAssertEqual(config.vocabularySize, 52000)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingEncoderBlockDecodesAsNil() throws {
        let noEncoderJSON = """
            {
                "model_type": "nemotron_parse",
                "decoder": {
                    "d_model": 1024,
                    "decoder_layers": 10,
                    "decoder_attention_heads": 16,
                    "vocab_size": 52000
                }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(noEncoderJSON.utf8))

        XCTAssertNil(config.encoderModelType)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testHeadCountMustDivideHiddenSize() throws {
        let badJSON = validJSON.replacingOccurrences(
            of: "\"decoder_attention_heads\": 16", with: "\"decoder_attention_heads\": 15")
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(badJSON.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testMissingDecoderBlockFailsDecoding() {
        let missingJSON = validJSON.replacingOccurrences(
            of: "\"d_model\": 1024,", with: "")

        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                SharpAINemotronParseConfiguration.self, from: Data(missingJSON.utf8)))
    }

    /// Without `encoder.args`, the C-RADIOv2-H defaults apply: 4 class tokens, summaries from 0, 1, 2.
    func testEncoderDefaultsWithoutArgsMatchRadioViTH() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(validJSON.utf8))

        XCTAssertEqual(config.encoder, SharpAINemotronParseEncoderShape.radioViTH)
        XCTAssertEqual(config.encoder.numClsTokens, 4)
        XCTAssertEqual(config.encoder.summaryIndices, [0, 1, 2])
        XCTAssertEqual(config.encoder.numSkipTokens, 8)
        XCTAssertEqual(config.encoder.summaryDim, 3840)
        XCTAssertEqual(config.encoder.cpeGridSide, 128)
    }

    /// The `encoder.args` block of nvidia/NVIDIA-Nemotron-Parse-v1.1 config.json (teachers and CPE).
    func testEncoderArgsDeriveClassAndSummaryLayout() throws {
        let json = """
            {
                "model_type": "nemotron_parse",
                "encoder": {
                    "model_type": "",
                    "patch_size": 16,
                    "args": {
                        "cpe_max_size": 2048,
                        "register_multiple": 8,
                        "cls_token_per_teacher": true,
                        "teachers": [
                            {"name": "clip", "use_summary": true},
                            {"name": "siglip", "use_summary": true},
                            {"name": "dino_v2", "use_summary": true},
                            {"name": "sam", "use_summary": false}
                        ]
                    }
                },
                "decoder": {
                    "d_model": 1024,
                    "decoder_layers": 10,
                    "decoder_attention_heads": 16,
                    "decoder_ffn_dim": 4096,
                    "encoder_ffn_dim": 4096,
                    "vocab_size": 52352
                },
                "decoder_start_token_id": 2,
                "pad_token_id": 1
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.encoder.numClsTokens, 4)
        XCTAssertEqual(config.encoder.summaryIndices, [0, 1, 2])
        XCTAssertEqual(config.encoder.numRegisters, 4)
        XCTAssertEqual(config.encoder.hiddenSize, 1280)
        XCTAssertEqual(config.encoder.layers, 32)
        XCTAssertEqual(config.ffnDim, 4096)
        XCTAssertEqual(config.decoderStartTokenId, 2)
        XCTAssertEqual(config.padTokenId, 1)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    /// Non-default `cpe_max_size` and `register_multiple` must be read from snake_case keys,
    /// not silently replaced by the RADIO defaults (2048 and 8).
    func testEncoderArgsCpeAndRegisterKeysAreRead() throws {
        let json = """
            {
                "model_type": "nemotron_parse",
                "encoder": {"args": {"cpe_max_size": 1024, "register_multiple": 16,
                    "cls_token_per_teacher": true,
                    "teachers": [{"name": "clip"}, {"name": "dino_v2"}]}},
                "decoder": {"d_model": 64, "decoder_layers": 1, "decoder_attention_heads": 4,
                    "vocab_size": 128}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.encoder.cpeMaxSize, 1024)
        XCTAssertEqual(config.encoder.cpeGridSide, 64)
        XCTAssertEqual(config.encoder.numClsTokens, 2)
        // register_multiple 16 with 2 class tokens gives 14 registers.
        XCTAssertEqual(config.encoder.numRegisters, 14)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testEncoderSummaryIndexOutOfRangeFailsValidation() throws {
        let json = """
            {
                "model_type": "nemotron_parse",
                "encoder": {"args": {"teachers": [{"name": "a"}, {"name": "b", "use_summary": true}],
                    "cls_token_per_teacher": false}},
                "decoder": {"d_model": 64, "decoder_layers": 1, "decoder_attention_heads": 4,
                    "vocab_size": 128}
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAINemotronParseConfiguration.self, from: Data(json.utf8))

        // cls_token_per_teacher false gives 1 class token, so summary index 1 is out of range.
        XCTAssertEqual(config.encoder.numClsTokens, 1)
        XCTAssertThrowsError(try config.validateModelConfiguration())
    }
}
