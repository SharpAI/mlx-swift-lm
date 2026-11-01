// SharpAI ParamsJson registration tests. The creator validates, then refuses a standalone model.
// UNVERIFIED: written without compiling. Run it before relying on it.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIParamsJsonRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForParamsJsonKey() {
        XCTAssertNotNil(SharpAIParamsJsonOverlay.creators["params.json"])
    }

    func testValidConfigFailsWithEagleRequiresTargetModel() throws {
        guard let creator = SharpAIParamsJsonOverlay.creators["params.json"] else {
            XCTFail("params.json creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data(SharpAIParamsJsonTestFixtures.tinyParamsJSON.utf8))) {
            error in
            XCTAssertEqual(
                error as? SharpAIParamsJsonError, .eagleRequiresTargetModel("params.json"))
        }
    }

    func testInvalidConfigFailsBeforeEagleError() throws {
        guard let creator = SharpAIParamsJsonOverlay.creators["params.json"] else {
            XCTFail("params.json creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data("{}".utf8))) { error in
            XCTAssertNotEqual(
                error as? SharpAIParamsJsonError, .eagleRequiresTargetModel("params.json"))
        }
    }

    func testOutOfRangeTopKFailsValidationBeforeEagleError() throws {
        guard let creator = SharpAIParamsJsonOverlay.creators["params.json"] else {
            XCTFail("params.json creator missing")
            return
        }
        let json = SharpAIParamsJsonTestFixtures.tinyParamsJSON.replacingOccurrences(
            of: "\"num_experts_per_tok\": 2", with: "\"num_experts_per_tok\": 9")
        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIParamsJsonError.invalidConfiguration = error else {
                XCTFail("expected invalidConfiguration, got \(error)")
                return
            }
        }
    }

    func testErrorDescriptionMentionsModelType() {
        let description =
            SharpAIParamsJsonError.eagleRequiresTargetModel("params.json").errorDescription ?? ""
        XCTAssertTrue(description.contains("params.json"))
    }
}
