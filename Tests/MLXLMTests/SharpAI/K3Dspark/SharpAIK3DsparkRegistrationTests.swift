// SharpAI fork overlay tests for the k3_dspark creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIK3DsparkRegistrationTests: XCTestCase {

    private let validJSON = """
        {
          "model_type": "k3_dspark",
          "architectures": ["K3DSparkModel"],
          "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
          "num_attention_heads": 2, "q_lora_rank": 8, "kv_lora_rank": 4,
          "qk_nope_head_dim": 4, "qk_rope_head_dim": 2, "v_head_dim": 4,
          "vocab_size": 50, "num_target_layers": 2, "target_hidden_size": 6,
          "target_layer_ids": [0, 1], "markov_rank": 4
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIK3DsparkOverlay.creators["k3_dspark"])
    }

    func testValidConfigFailsBecauseTargetIsMissing() throws {
        let creator = try XCTUnwrap(SharpAIK3DsparkOverlay.creators["k3_dspark"])

        XCTAssertThrowsError(try creator(Data(validJSON.utf8))) { error in
            guard case SharpAIK3DsparkError.targetNotImplemented(let type) = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
            XCTAssertEqual(type, "k3_dspark")
        }
    }

    func testInvalidConfigFailsBeforeTargetCheck() throws {
        let creator = try XCTUnwrap(SharpAIK3DsparkOverlay.creators["k3_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "\"q_lora_rank\": 8", with: "\"q_lora_rank\": null")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }

    func testWrongModelTypeFailsBeforeTargetCheck() throws {
        let creator = try XCTUnwrap(SharpAIK3DsparkOverlay.creators["k3_dspark"])
        let json = validJSON.replacingOccurrences(
            of: "\"model_type\": \"k3_dspark\"", with: "\"model_type\": \"llama\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case ModelFactoryError.invalidConfiguration = error else {
                XCTFail("config validation should fail first: \(error)")
                return
            }
        }
    }
}
