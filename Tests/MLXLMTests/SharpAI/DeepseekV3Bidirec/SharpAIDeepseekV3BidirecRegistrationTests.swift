// SharpAI fork overlay tests for the deepseek_v3_bidirec creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDeepseekV3BidirecRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "deepseek_v3_bidirec",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 2,
            "v_head_dim": 4
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDeepseekV3BidirecOverlay.creators["deepseek_v3_bidirec"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIDeepseekV3BidirecOverlay.creators["deepseek_v3_bidirec"]
        else {
            XCTFail("deepseek_v3_bidirec creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAIDeepseekV3BidirecModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIDeepseekV3BidirecOverlay.creators["deepseek_v3_bidirec"]
        else {
            XCTFail("deepseek_v3_bidirec creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 8 }".utf8)))
    }
}
