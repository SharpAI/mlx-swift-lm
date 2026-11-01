// SharpAI fork overlay tests for the vaultgemma creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIVaultgemmaRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIVaultgemmaOverlay.creators["vaultgemma"])
    }

    func testValidTinyConfigBuildsModel() throws {
        guard let creator = SharpAIVaultgemmaOverlay.creators["vaultgemma"] else {
            XCTFail("vaultgemma creator missing")
            return
        }
        let model = try creator(Data(SharpAIVaultgemmaTestFixture.tinyJSON().utf8))
        XCTAssertTrue(model is SharpAIVaultgemmaModel)
    }

    func testNonDefaultRopeFailsBeforeConstruction() throws {
        guard let creator = SharpAIVaultgemmaOverlay.creators["vaultgemma"] else {
            XCTFail("vaultgemma creator missing")
            return
        }
        let json = #"{"rope_parameters": {"rope_type": "linear", "factor": 2.0}}"#

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
