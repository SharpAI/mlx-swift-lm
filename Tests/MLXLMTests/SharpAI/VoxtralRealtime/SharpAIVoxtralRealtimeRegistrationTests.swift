// SharpAI fork overlay tests for the voxtral_realtime creator. No network.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIVoxtralRealtimeRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAIVoxtralRealtimeOverlay.creators["voxtral_realtime"])
    }

    func testValidConfigBuildsModel() throws {
        let creator = try XCTUnwrap(SharpAIVoxtralRealtimeOverlay.creators["voxtral_realtime"])
        let model = try creator(Data(SharpAIVoxtralRealtimeTestConfig.tiny().utf8))
        XCTAssertTrue(model is SharpAIVoxtralRealtimeForConditionalGeneration)
    }

    func testInvalidConfigThrowsInCreator() throws {
        let creator = try XCTUnwrap(SharpAIVoxtralRealtimeOverlay.creators["voxtral_realtime"])
        // 2 heads with 3 kv heads does not divide.
        let json = SharpAIVoxtralRealtimeTestConfig.tiny(numKvHeads: 3)
        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
