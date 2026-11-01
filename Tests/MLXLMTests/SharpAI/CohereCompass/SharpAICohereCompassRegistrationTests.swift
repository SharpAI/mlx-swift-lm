// SharpAI fork overlay tests for the cohere_compass creator. Not part of upstream.

import Foundation
import MLX
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAICohereCompassRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "cohere_compass",
            "architectures": ["CohereCompassForConditionalGeneration"]
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAICohereCompassOverlay.creators["cohere_compass"])
    }

    func testValidConfigFailsUntilModelIsImplemented() throws {
        guard let creator = SharpAICohereCompassOverlay.creators["cohere_compass"] else {
            XCTFail("cohere_compass creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAICohereCompassError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "cohere_compass")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAICohereCompassOverlay.creators["cohere_compass"] else {
            XCTFail("cohere_compass creator missing")
            return
        }
        let json = """
            {
                "model_type": "cohere_compass",
                "architectures": ["SomeOtherForConditionalGeneration"]
            }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            if case SharpAICohereCompassError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Tiny text_config with the reference layout: sliding layers with RoPE, a full layer with
    /// no RoPE (rope_parameters entry is null). Sequence length 6 exceeds sliding_window 4, so
    /// the window mask is exercised.
    private let tinyTextConfigJSON = """
        {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "layer_norm_eps": 1e-05,
            "logit_scale": 0.25,
            "attention_bias": false,
            "sliding_window": 4,
            "hidden_act": "silu",
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
            ],
            "rope_parameters": {
                "sliding_attention": { "rope_theta": 50000, "rope_type": "default" },
                "full_attention": null
            }
        }
        """

    func testTinyTextForwardPass() throws {
        let config = try JSONDecoder.json5().decode(
            SharpAICohereCompassTextConfiguration.self, from: Data(tinyTextConfigJSON.utf8))
        XCTAssertEqual(config.layerRopeTheta.count, 4)
        XCTAssertEqual(config.layerRopeTheta[0], Float(50000))
        XCTAssertNil(config.layerRopeTheta[3])

        let model = SharpAICohereCompassTextModel(config)

        let tokens = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(1, 6)
        let logits = model(tokens, cache: nil)
        XCTAssertEqual(logits.shape, [1, 6, 64])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })

        // Causality: changing the last token must not change logits at earlier positions.
        let changed = MLXArray([Int32(1), 2, 3, 4, 5, 7]).reshaped(1, 6)
        let logitsChanged = model(changed, cache: nil)
        let before = logits[0..., 0 ..< 5, 0...].asArray(Float.self)
        let after = logitsChanged[0..., 0 ..< 5, 0...].asArray(Float.self)
        XCTAssertEqual(before.count, after.count)
        XCTAssertTrue(zip(before, after).allSatisfy { abs($0 - $1) < 1e-4 })
    }
}
