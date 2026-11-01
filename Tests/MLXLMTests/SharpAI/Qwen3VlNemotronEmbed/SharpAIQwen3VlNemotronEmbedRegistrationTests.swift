import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIQwen3VlNemotronEmbedRegistrationTests: XCTestCase {

    func testOverlayRegistersModelTypeKey() throws {
        let creator = try XCTUnwrap(
            SharpAIQwen3VlNemotronEmbedOverlay.creators["qwen3_vl_nemotron_embed"])

        let model = try creator(SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigData)

        // Output width is the text hidden size (16 in the tiny fixture), not the local `embed_dim` key (8).
        // The reference checkpoint has no `embed_dim` and no projection head; last_hidden_state is returned as-is.
        XCTAssertEqual(model.embedDim, 16)
    }

    func testOverlayCreatorRunsValidation() throws {
        let creator = try XCTUnwrap(
            SharpAIQwen3VlNemotronEmbedOverlay.creators["qwen3_vl_nemotron_embed"])
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "\"embed_dim\": 0,")

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }

    func testOverlayHasNoUnrelatedKeys() {
        XCTAssertEqual(
            Set(SharpAIQwen3VlNemotronEmbedOverlay.creators.keys), ["qwen3_vl_nemotron_embed"])
    }
}
