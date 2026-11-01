import Foundation
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIPhi4SiglipRegistrationTests: XCTestCase {

    func testCreatorIsKeyedByModelType() throws {
        XCTAssertNotNil(SharpAIPhi4SiglipOverlay.creators["phi4-siglip"])
        XCTAssertNil(SharpAIPhi4SiglipOverlay.creators["phi4"])
    }

    func testCreatorBuildsModelFromTinyConfig() throws {
        let creator = try XCTUnwrap(SharpAIPhi4SiglipOverlay.creators["phi4-siglip"])

        let model = try creator(Data(SharpAIPhi4SiglipTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIPhi4SiglipModel)
    }

    func testCreatorRejectsInvalidConfig() throws {
        let creator = try XCTUnwrap(SharpAIPhi4SiglipOverlay.creators["phi4-siglip"])
        let json = SharpAIPhi4SiglipTestFixtures.tinyJSON
            .replacingOccurrences(of: "\"rope_scaling\": null,", with: "\"rope_scaling\": {\"type\": \"linear\"},")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
