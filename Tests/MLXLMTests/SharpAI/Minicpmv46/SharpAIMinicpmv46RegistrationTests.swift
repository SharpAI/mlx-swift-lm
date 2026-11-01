// SharpAI fork overlay tests for the minicpmv4_6 creator. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIMinicpmv46RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMinicpmv46Overlay.creators["minicpmv4_6"])
    }

    func testCreatorBuildsModelFromTinyConfig() throws {
        guard let creator = SharpAIMinicpmv46Overlay.creators["minicpmv4_6"] else {
            XCTFail("minicpmv4_6 creator missing")
            return
        }
        let model = try creator(Data(SharpAIMinicpmv46SanitizeTests.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIMinicpmv46)
    }

    func testEmptyModelTypeFailsValidationBeforeConstruction() throws {
        guard let creator = SharpAIMinicpmv46Overlay.creators["minicpmv4_6"] else {
            XCTFail("minicpmv4_6 creator missing")
            return
        }
        let missingText = "{ \"model_type\": \"minicpmv4_6\" }"

        XCTAssertThrowsError(try creator(Data(missingText.utf8)))
    }
}
