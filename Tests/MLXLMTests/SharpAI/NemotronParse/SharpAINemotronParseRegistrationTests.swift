// SharpAI fork overlay tests for the nemotron_parse creator. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAINemotronParseRegistrationTests: XCTestCase {

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

    func testCreatorKeyIsRegistered() {
        XCTAssertNotNil(SharpAINemotronParseOverlay.creators["nemotron_parse"])
    }

    /// Previously expected `forwardNotImplemented`. The model now exists, so the creator returns it.
    func testValidConfigBuildsModel() throws {
        let creator = SharpAINemotronParseOverlay.creators["nemotron_parse"]!

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAINemotronParse)
    }

    func testInvalidConfigFailsValidationBeforeConstruction() {
        let badJSON = validJSON.replacingOccurrences(
            of: "\"decoder_attention_heads\": 16", with: "\"decoder_attention_heads\": 15")
        let creator = SharpAINemotronParseOverlay.creators["nemotron_parse"]!

        XCTAssertThrowsError(try creator(Data(badJSON.utf8))) { error in
            if case SharpAINemotronParseError.forwardNotImplemented = error {
                XCTFail("Validation should fail before construction")
            }
        }
    }

    /// Reduced-size tiny model: the encoder dimensions come from the optional keys (absent in the
    /// real checkpoint, which uses ViT-H). Image 64x64 gives 4x4 patches, 8 skip tokens (4 class +
    /// 4 registers), a 4x1 grid after the 1x4 conv2, and one summary token: 5 encoder tokens.
    func testTinyForwardPassShape() throws {
        let tinyJSON = """
            {
                "model_type": "nemotron_parse",
                "architectures": ["NemotronParseForConditionalGeneration"],
                "encoder": {
                    "model_type": "radio",
                    "hidden_size": 64,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "mlp_dim": 128
                },
                "decoder": {
                    "d_model": 64,
                    "decoder_layers": 1,
                    "decoder_attention_heads": 4,
                    "decoder_ffn_dim": 128,
                    "encoder_ffn_dim": 128,
                    "vocab_size": 128
                },
                "decoder_start_token_id": 2,
                "pad_token_id": 1
            }
            """
        let creator = SharpAINemotronParseOverlay.creators["nemotron_parse"]!
        let model = try creator(Data(tinyJSON.utf8))
        let nemotron = try XCTUnwrap(model as? SharpAINemotronParse)

        let pixels = MLXArray.zeros([1, 3, 64, 64], dtype: .float32)
        let states = try nemotron.encode(pixelValues: pixels)
        XCTAssertEqual(states.shape, [1, 5, 64])

        let tokens = MLXArray([Int32(2), 5, 7], [1, 3])
        let logits = try nemotron.logits(tokens, pixelValues: pixels)
        XCTAssertEqual(logits.shape, [1, 3, 128])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testTinyForwardRejectsNonPatchMultipleResolution() throws {
        let tinyJSON = """
            {
                "model_type": "nemotron_parse",
                "encoder": {"model_type": "radio", "hidden_size": 64,
                    "num_hidden_layers": 1, "num_attention_heads": 4, "mlp_dim": 128},
                "decoder": {"d_model": 64, "decoder_layers": 1,
                    "decoder_attention_heads": 4, "decoder_ffn_dim": 128, "vocab_size": 128}
            }
            """
        let creator = SharpAINemotronParseOverlay.creators["nemotron_parse"]!
        let nemotron = try XCTUnwrap(try creator(Data(tinyJSON.utf8)) as? SharpAINemotronParse)

        XCTAssertThrowsError(
            try nemotron.encode(pixelValues: MLXArray.zeros([1, 3, 60, 64], dtype: .float32)))
    }
}
