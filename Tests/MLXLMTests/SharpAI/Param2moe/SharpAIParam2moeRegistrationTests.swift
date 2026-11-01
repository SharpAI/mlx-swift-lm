// SharpAI fork overlay tests for the param2moe creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIParam2moeRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIParam2moeOverlay.creators["param2moe"])
    }

    func testValidTinyConfigBuildsModel() throws {
        guard let creator = SharpAIParam2moeOverlay.creators["param2moe"] else {
            XCTFail("param2moe creator missing")
            return
        }
        let model = try creator(Data(try SharpAIParam2moeTestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAIParam2moeModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIParam2moeOverlay.creators["param2moe"] else {
            XCTFail("param2moe creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 32 }".utf8)))
    }

    func testTopKOverflowFailsBeforeConstruction() throws {
        guard let creator = SharpAIParam2moeOverlay.creators["param2moe"] else {
            XCTFail("param2moe creator missing")
            return
        }
        let json = try SharpAIParam2moeTestFixture.tinyJSON(["num_experts_per_tok": 9])

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIParam2moeError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
