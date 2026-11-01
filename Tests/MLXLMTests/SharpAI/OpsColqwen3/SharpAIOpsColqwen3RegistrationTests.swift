import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIOpsColqwen3RegistrationTests: XCTestCase {

    func testOverlayRegistersOpsColqwen3Key() throws {
        let creator = try XCTUnwrap(SharpAIOpsColqwen3Overlay.creators["ops_colqwen3"])

        let model = try creator(SharpAIOpsColqwen3TestFixtures.tinyConfigData)

        XCTAssertEqual(model.dims, 8)
    }

    func testOverlayCreatorRunsValidation() throws {
        let creator = try XCTUnwrap(SharpAIOpsColqwen3Overlay.creators["ops_colqwen3"])
        let json = SharpAIOpsColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"dims\": 8,", with: "\"dims\": 0,")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testOverlayHasNoUnrelatedKeys() {
        XCTAssertEqual(Set(SharpAIOpsColqwen3Overlay.creators.keys), ["ops_colqwen3"])
    }
}
