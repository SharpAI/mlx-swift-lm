// SharpAI fork known-issue marker for the sarvam_mla forward model. Not part of upstream.

import Foundation
import XCTest

@testable import MLXLLM

final class SharpAISarvamMlaModelTests: XCTestCase {

    // UNVERIFIED: no forward model exists yet. Needs modeling_sarvam_moe.py (MLA
    // attention, use_qk_norm placement, router with expert bias). Once the model
    // class exists, replace this with a tiny-shape forward pass.
    // Known issue: a valid tiny config stops at the unsupported forward model.
    func testTinyConfigStopsAtUnsupportedForwardModel() {
        guard let creator = SharpAISarvamMlaOverlay.creators["sarvam_mla"] else {
            XCTFail("sarvam_mla creator missing")
            return
        }
        XCTAssertThrowsError(
            try creator(Data(SharpAISarvamMlaTestFixture.tinyJSON().utf8))
        ) { error in
            guard case SharpAISarvamMlaError.unsupported = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }
}
