// SharpAI fork overlay tests for the mimo_v2 creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMimoV2RegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIMimoV2Overlay.creators["mimo_v2"])
    }

    func testValidTinyConfigBuildsModel() throws {
        guard let creator = SharpAIMimoV2Overlay.creators["mimo_v2"] else {
            XCTFail("mimo_v2 creator missing")
            return
        }
        let model = try creator(Data(SharpAIMimoV2TestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAIMimoV2Model)
    }

    func testInvalidConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIMimoV2Overlay.creators["mimo_v2"] else {
            XCTFail("mimo_v2 creator missing")
            return
        }

        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 32 }".utf8)))
    }

    func testSharedExpertConfigFailsBeforeConstruction() throws {
        guard let creator = SharpAIMimoV2Overlay.creators["mimo_v2"] else {
            XCTFail("mimo_v2 creator missing")
            return
        }

        XCTAssertThrowsError(
            try creator(Data(SharpAIMimoV2TestFixture.tinyJSON(sharedExperts: 1).utf8))
        ) { error in
            guard case SharpAIMimoV2Error.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
