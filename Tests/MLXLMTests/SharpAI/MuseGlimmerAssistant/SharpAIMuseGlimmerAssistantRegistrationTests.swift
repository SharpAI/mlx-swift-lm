// SharpAI fork overlay tests for the muse_glimmer_assistant creator and key sanitize.
// Not part of upstream. Offline only: inline JSON, no checkpoint.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIMuseGlimmerAssistantRegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "model_type": "muse_glimmer_assistant",
          "architectures": ["MuseGlimmerAssistantModel"],
          "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "layer_types": ["full_attention", "sliding_attention"],
          "sliding_window": 4,
          "target_layer_ids": [0, 1],
          "block_size": 4, "mask_token_id": 9
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMuseGlimmerAssistantOverlay.creators["muse_glimmer_assistant"])
    }

    func testValidConfigFailsBecauseDrafterIsNotImplemented() throws {
        let creator = try XCTUnwrap(
            SharpAIMuseGlimmerAssistantOverlay.creators["muse_glimmer_assistant"])

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIMuseGlimmerAssistantError.drafterNotImplemented(let type) = error
            else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "muse_glimmer_assistant")
        }
    }

    func testInvalidConfigFailsBeforeDrafterCheck() throws {
        let creator = try XCTUnwrap(
            SharpAIMuseGlimmerAssistantOverlay.creators["muse_glimmer_assistant"])
        let json = validJSON.replacingOccurrences(
            of: "\"target_layer_ids\": [0, 1]", with: "\"target_layer_ids\": []")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeDrafterCheck() throws {
        let creator = try XCTUnwrap(
            SharpAIMuseGlimmerAssistantOverlay.creators["muse_glimmer_assistant"])
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"muse_glimmer_assistant\"", with: "\"model_type\": \"llama\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }

    // MARK: - Sanitize

    func testSanitizeDropsRotaryBuffersAndKeepsHFNames() {
        let weights: [String: MLXArray] = [
            "fc.weight": MLXArray.zeros([2, 2]),
            "layers.0.self_attn.q_proj.weight": MLXArray.zeros([2, 2]),
            "norm.weight": MLXArray.zeros([2]),
            "layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([1]),
        ]

        let sanitized = SharpAIMuseGlimmerAssistantSanitizer.sanitize(weights)

        XCTAssertEqual(
            Set(sanitized.keys),
            ["fc.weight", "layers.0.self_attn.q_proj.weight", "norm.weight"])
    }
}
