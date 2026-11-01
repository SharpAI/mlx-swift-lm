// SharpAI fork overlay tests for the modernbert creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIModernbertRegistrationTests: XCTestCase {

    private let validJSON = """
        {
            "model_type": "modernbert",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "norm_eps": 1e-5
        }
        """

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIModernbertOverlay.creators["modernbert"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIModernbertOverlay.creators["modernbert"] else {
            XCTFail("modernbert creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAIModernbertModel)
    }

    func testInvalidConfigThrowsInCreator() throws {
        guard let creator = SharpAIModernbertOverlay.creators["modernbert"] else {
            XCTFail("modernbert creator missing")
            return
        }
        let invalid = validJSON.replacingOccurrences(
            of: "\"num_attention_heads\": 2", with: "\"num_attention_heads\": 3")

        XCTAssertThrowsError(try creator(Data(invalid.utf8)))
    }
}
