// SharpAI fork overlay tests for the domynedge creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDomynedgeRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIDomynedgeOverlay.creators["domynedge"])
    }

    func testValidConfigBuildsDomynedgeModel() throws {
        let creator = try XCTUnwrap(SharpAIDomynedgeOverlay.creators["domynedge"])
        let model = try creator(Data(SharpAIDomynedgeTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIDomynedgeModel)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        let creator = try XCTUnwrap(SharpAIDomynedgeOverlay.creators["domynedge"])
        let json = SharpAIDomynedgeTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"window_sizes\": [[-1, 0], [4, 0], [4, 0], [-1, 0]]",
            with: "\"window_sizes\": [[-1, 0]]")

        XCTAssertThrowsError(try creator(Data(json.utf8))) { error in
            guard case SharpAIDomynedgeError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
