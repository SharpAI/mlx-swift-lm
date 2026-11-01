// SharpAI fork overlay tests for the sarvam_mla creator. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAISarvamMlaRegistrationTests: XCTestCase {

    func testCreatorIsRegisteredForModelType() {
        XCTAssertNotNil(SharpAISarvamMlaOverlay.creators["sarvam_mla"])
    }

    func testFixtureConfigReachesUnsupportedStop() {
        guard let creator = SharpAISarvamMlaOverlay.creators["sarvam_mla"] else {
            XCTFail("sarvam_mla creator missing")
            return
        }
        // Decode and validation pass; the forward model is the stop point.
        XCTAssertThrowsError(
            try creator(Data(SharpAISarvamMlaTestFixture.sarvam105bJSON().utf8))
        ) { error in
            guard case SharpAISarvamMlaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testInvalidConfigFailsBeforeUnsupportedStop() {
        guard let creator = SharpAISarvamMlaOverlay.creators["sarvam_mla"] else {
            XCTFail("sarvam_mla creator missing")
            return
        }
        XCTAssertThrowsError(try creator(Data("{ \"hidden_size\": 32 }".utf8))) { error in
            if case SharpAISarvamMlaError.unsupported = error {
                XCTFail("invalid config should fail in decode, not reach unsupported stop")
            }
        }
    }

    func testInconsistentConfigFailsValidation() {
        guard let creator = SharpAISarvamMlaOverlay.creators["sarvam_mla"] else {
            XCTFail("sarvam_mla creator missing")
            return
        }
        XCTAssertThrowsError(
            try creator(Data(SharpAISarvamMlaTestFixture.tinyJSON(headDim: 48).utf8))
        ) { error in
            guard case SharpAISarvamMlaError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
