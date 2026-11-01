// SharpAI fork overlay tests for the laguna creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAILagunaRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAILagunaOverlay.creators["laguna"])
    }

    func testValidTinyConfigBuildsModel() throws {
        guard let creator = SharpAILagunaOverlay.creators["laguna"] else {
            XCTFail("laguna creator missing")
            return
        }
        let model = try creator(Data(SharpAILagunaTestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAILagunaModel)
    }

    func testDFlashDrafterFailsBeforeConstruction() throws {
        guard let creator = SharpAILagunaOverlay.creators["laguna"] else {
            XCTFail("laguna creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(SharpAILagunaTestFixture.dflashJSON.utf8))) {
            error in
            guard case SharpAILagunaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAILagunaOverlay.creators["laguna"] else {
            XCTFail("laguna creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 32 }".utf8)))
    }
}
