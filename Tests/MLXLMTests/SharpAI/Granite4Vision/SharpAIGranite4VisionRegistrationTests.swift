// SharpAI fork overlay tests for the granite4_vision creator. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXVLM
import XCTest

final class SharpAIGranite4VisionRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "granite4_vision",
            "text_config": {
                "model_type": "granite",
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "intermediate_size": 512,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-5,
                "vocab_size": 1024,
                "logits_scaling": 1.0,
                "attention_multiplier": 1.0,
                "embedding_multiplier": 1.0,
                "residual_multiplier": 1.0,
                "max_position_embeddings": 4096,
                "attention_bias": false,
                "mlp_bias": false,
                "rope_theta": 10000.0,
                "tie_word_embeddings": true
            },
            "vision_config": {}
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIGranite4VisionOverlay.creators["granite4_vision"])
    }

    func testValidConfigFailsUntilForwardIsImplemented() throws {
        guard let creator = SharpAIGranite4VisionOverlay.creators["granite4_vision"] else {
            XCTFail("granite4_vision creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIGranite4VisionError.forwardNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "granite4_vision")
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIGranite4VisionOverlay.creators["granite4_vision"] else {
            XCTFail("granite4_vision creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 64 }".utf8))) { error in
            if case SharpAIGranite4VisionError.forwardNotImplemented = error {
                XCTFail("config validation should fail first")
            }
        }
    }

    /// Tiny random-weight forward pass of the granite4_vision text stack.
    ///
    /// Covers the nested `text_config` decode and the GraniteModel forward with the
    /// granite4 multipliers. The vision tower, window Q-Former and deepstack injection are
    /// NOT covered here; they are still unimplemented (see the creator's forwardNotImplemented).
    func testTinyTextForwardPass() throws {
        let configuration = try JSONDecoder.json5().decode(
            SharpAIGranite4VisionConfiguration.self, from: Data(validJSON.utf8))
        try configuration.validateModelConfiguration()

        let model = GraniteModel(configuration.textConfiguration)
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])

        let logits = model(ids, cache: nil)
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 4, 1024])
        XCTAssertTrue(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
