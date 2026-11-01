import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIColqwen3RegistrationTests: XCTestCase {

    func testOverlayRegistersColqwen3Key() throws {
        let creator = try XCTUnwrap(SharpAIColqwen3Overlay.creators["colqwen3"])

        let model = try creator(SharpAIColqwen3TestFixtures.tinyConfigData)

        XCTAssertEqual(model.embedDim, 8)
    }

    func testOverlayCreatorRunsValidation() throws {
        let creator = try XCTUnwrap(SharpAIColqwen3Overlay.creators["colqwen3"])
        let json = SharpAIColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "\"embed_dim\": 0,")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testOverlayHasNoUnrelatedKeys() {
        XCTAssertEqual(Set(SharpAIColqwen3Overlay.creators.keys), ["colqwen3"])
    }
}
