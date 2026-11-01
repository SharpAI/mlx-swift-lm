// SharpAI fork overlay tests for the openai_privacy_filter creator. Not part of upstream.

import Foundation
import XCTest

@testable import MLXLLM

final class SharpAIOpenaiPrivacyFilterRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "openai_privacy_filter",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 8,
            "head_dim": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "sliding_window": 2,
            "id2label": {"0": "O", "1": "B-secret"}
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIOpenaiPrivacyFilterOverlay.creators["openai_privacy_filter"])
    }

    func testValidConfigBuildsTokenClassifier() throws {
        guard let creator = SharpAIOpenaiPrivacyFilterOverlay.creators["openai_privacy_filter"]
        else {
            XCTFail("openai_privacy_filter creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAIOpenaiPrivacyFilterModel)
    }

    func testInvalidConfigThrowsInCreator() throws {
        guard let creator = SharpAIOpenaiPrivacyFilterOverlay.creators["openai_privacy_filter"]
        else {
            XCTFail("openai_privacy_filter creator missing")
            return
        }
        let invalid = validJSON.replacingOccurrences(
            of: "\"num_attention_heads\": 4", with: "\"num_attention_heads\": 3")

        XCTAssertThrowsError(try creator(Data(invalid.utf8)))
    }
}
