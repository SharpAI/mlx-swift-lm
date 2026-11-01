// SharpAI fork overlay tests for the openvla creator. Not part of upstream.
//
// Only the failure paths and the key set are checked. A valid creator call would build the full
// OpenVLA-7B vision stack, so it is left out of unit tests.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIOpenvlaRegistrationTests: XCTestCase {

    func testRegistersOnlyTheOpenVLAKey() {
        XCTAssertEqual(Array(SharpAIOpenvlaOverlay.creators.keys), ["openvla"])
    }

    func testCreatorRejectsUnsupportedBackboneBeforeBuildingModel() throws {
        let creator = try XCTUnwrap(SharpAIOpenvlaOverlay.creators["openvla"])
        let json = """
            {
              "model_type": "openvla",
              "vision_backbone_id": "clip-vit-l-336px"
            }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
