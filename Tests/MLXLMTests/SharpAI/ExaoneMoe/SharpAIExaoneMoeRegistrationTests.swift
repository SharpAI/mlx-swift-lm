// SharpAI fork overlay tests for the exaone_moe creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIExaoneMoeRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "exaone_moe",
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIExaoneMoeOverlay.creators["exaone_moe"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIExaoneMoeOverlay.creators["exaone_moe"] else {
            XCTFail("exaone_moe creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is LLMModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIExaoneMoeOverlay.creators["exaone_moe"] else {
            XCTFail("exaone_moe creator missing")
            return
        }
        let json = """
            { "hidden_size": 16 }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testUnsupportedActivationFailsAtValidation() throws {
        guard let creator = SharpAIExaoneMoeOverlay.creators["exaone_moe"] else {
            XCTFail("exaone_moe creator missing")
            return
        }
        let json = validJSON.replacingOccurrences(
            of: "\"head_dim\": 8", with: "\"head_dim\": 8, \"hidden_act\": \"gelu\"")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIExaoneMoeError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
