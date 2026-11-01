// SharpAI fork overlay tests for the granitemoe_swa creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIGranitemoeSwaRegistrationTests: XCTestCase {

    private let validJSON = SharpAIGranitemoeSwaTestFixtures.tinyJSON

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIGranitemoeSwaOverlay.creators["granitemoe_swa"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIGranitemoeSwaOverlay.creators["granitemoe_swa"] else {
            XCTFail("granitemoe_swa creator missing")
            return
        }

        let model = try creator(Data(validJSON.utf8))

        XCTAssertTrue(model is SharpAIGranitemoeSwaModel)
    }

    func testMissingRequiredFieldFailsDecoding() {
        guard let creator = SharpAIGranitemoeSwaOverlay.creators["granitemoe_swa"] else {
            XCTFail("granitemoe_swa creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(#"{ "hidden_size": 16 }"#.utf8)))
    }

    func testInvalidConfigFailsValidationBeforeConstruction() throws {
        guard let creator = SharpAIGranitemoeSwaOverlay.creators["granitemoe_swa"] else {
            XCTFail("granitemoe_swa creator missing")
            return
        }
        // Layer 0 slides but no sliding_window is set.
        let json = validJSON.replacingOccurrences(
            of: "\"sliding_window\": 2,", with: "")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
