// SharpAI fork overlay tests for the paddleocr_vl creator. Not part of upstream.

import Foundation
import MLXLMCommon
@testable import MLXVLM
import XCTest

final class SharpAIPaddleocrVlRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIPaddleocrVlOverlay.creators["paddleocr_vl"])
    }

    func testCreatorBuildsModelFromTinyConfig() throws {
        let creator = try XCTUnwrap(SharpAIPaddleocrVlOverlay.creators["paddleocr_vl"])
        let model = try creator(Data(SharpAIPaddleocrVlConfigurationTests.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIPaddleocrVl)
    }

    func testMissingTextConfigFailsBeforeConstruction() throws {
        let creator = try XCTUnwrap(SharpAIPaddleocrVlOverlay.creators["paddleocr_vl"])

        XCTAssertThrowsError(
            try creator(Data("{ \"model_type\": \"paddleocr_vl\" }".utf8)))
    }
}
