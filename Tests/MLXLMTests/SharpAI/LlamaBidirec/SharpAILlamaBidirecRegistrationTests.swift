// SharpAI fork overlay tests for the llama_bidirec creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILlamaBidirecRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "llama_bidirec",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-6
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAILlamaBidirecOverlay.creators["llama_bidirec"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAILlamaBidirecOverlay.creators["llama_bidirec"] else {
            XCTFail("llama_bidirec creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAILlamaBidirecModel)
    }

    func testInvalidConfigThrowsInCreator() throws {
        guard let creator = SharpAILlamaBidirecOverlay.creators["llama_bidirec"] else {
            XCTFail("llama_bidirec creator missing")
            return
        }
        let invalid = validJSON.replacingOccurrences(
            of: "\"num_attention_heads\": 2", with: "\"num_attention_heads\": 0")

        XCTAssertThrowsError(try creator(Data(invalid.utf8)))
    }
}
