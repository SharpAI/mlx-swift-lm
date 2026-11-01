// SharpAI fork overlay: moondream3 creator tests. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIMoondream3RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMoondream3Overlay.creators["moondream3"])
    }

    func testValidConfigBuildsModel() throws {
        guard let creator = SharpAIMoondream3Overlay.creators["moondream3"] else {
            return XCTFail("moondream3 creator missing")
        }

        let model = try creator(Data(SharpAIMoondream3TestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIMoondream3Model)
    }

    func testWrongModelTypeFailsBeforeConstruction() {
        guard let creator = SharpAIMoondream3Overlay.creators["moondream3"] else {
            return XCTFail("moondream3 creator missing")
        }

        XCTAssertThrowsError(
            try creator(Data(#"{"model_type": "moondream2"}"#.utf8)))
    }
}
