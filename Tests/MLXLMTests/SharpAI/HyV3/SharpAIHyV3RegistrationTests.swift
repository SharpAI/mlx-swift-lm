// SharpAI fork overlay tests for the hy_v3 creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIHyV3RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIHyV3Overlay.creators["hy_v3"])
    }

    func testValidTinyConfigBuildsModel() throws {
        guard let creator = SharpAIHyV3Overlay.creators["hy_v3"] else {
            XCTFail("hy_v3 creator missing")
            return
        }
        let model = try creator(Data(SharpAIHyV3TestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAIHyV3Model)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIHyV3Overlay.creators["hy_v3"] else {
            XCTFail("hy_v3 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 32 }".utf8)))
    }

    func testUnsupportedRouterFailsBeforeConstruction() throws {
        guard let creator = SharpAIHyV3Overlay.creators["hy_v3"] else {
            XCTFail("hy_v3 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data(SharpAIHyV3TestFixture.tinyJSON(sigmoid: false).utf8))) {
            error in
            guard case SharpAIHyV3Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
