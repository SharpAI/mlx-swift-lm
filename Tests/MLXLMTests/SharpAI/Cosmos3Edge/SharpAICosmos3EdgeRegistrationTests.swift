// SharpAI fork overlay tests for the cosmos3_edge creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICosmos3EdgeRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "cosmos3_edge",
            "architectures": ["Cosmos3EdgeForConditionalGeneration"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAICosmos3EdgeOverlay.creators["cosmos3_edge"])
    }

    func testValidConfigFailsUntilModelIsImplemented() throws {
        guard let creator = SharpAICosmos3EdgeOverlay.creators["cosmos3_edge"] else {
            XCTFail("cosmos3_edge creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAICosmos3EdgeError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "cosmos3_edge")
        }
    }

    func testWrongModelTypeFailsBeforeConstruction() throws {
        guard let creator = SharpAICosmos3EdgeOverlay.creators["cosmos3_edge"] else {
            XCTFail("cosmos3_edge creator missing")
            return
        }
        let json = """
            { "model_type": "llama" }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            if case SharpAICosmos3EdgeError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Tiny text_config with random weights. head_dim 8 gives 4 rotary slots, and mrope_section
    /// [2, 1, 1] sums to 4. The vision path is not part of this test.
    private let tinyJSON = """
        {
            "model_type": "cosmos3_edge",
            "architectures": ["Cosmos3EdgeForConditionalGeneration"],
            "text_config": {
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "hidden_act": "relu2",
                "rms_norm_eps": 1e-5,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000,
                    "mrope_section": [2, 1, 1]
                }
            }
        }
        """

    func testTinyForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(tinyJSON.utf8))
        let model = try SharpAICosmos3EdgeTextForCausalLM(config.textConfiguration)

        let tokens = MLXArray([Int32(1), 5, 9, 3, 7]).reshaped(1, 5)
        let logits = model(tokens)

        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self), "NaN logits")
    }

    func testTinyForwardIsCausal() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(tinyJSON.utf8))
        let model = try SharpAICosmos3EdgeTextForCausalLM(config.textConfiguration)

        // Changing only the last token must leave the logits of earlier positions unchanged.
        let a = model(MLXArray([Int32(1), 5, 9, 3, 7]).reshaped(1, 5))
        let b = model(MLXArray([Int32(1), 5, 9, 3, 2]).reshaped(1, 5))

        XCTAssertTrue(
            allClose(a[0..., ..<4], b[0..., ..<4], rtol: 1e-4, atol: 1e-4)
                .item(Bool.self))
    }

    func testTinyConfigRejectsMissingRopeParameters() throws {
        let json = """
            {
                "model_type": "cosmos3_edge",
                "text_config": { "head_dim": 8, "num_attention_heads": 4, "num_key_value_heads": 2 }
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAICosmos3EdgeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try SharpAICosmos3EdgeTextForCausalLM(config.textConfiguration))
    }
}
