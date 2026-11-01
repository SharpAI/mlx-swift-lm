import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIColqwen3ConfigTests: XCTestCase {

    func testDecodesBackboneAndEmbedDim() throws {
        let config = try SharpAIColqwen3TestFixtures.tinyConfiguration()

        XCTAssertEqual(config.embedDim, 8)
        XCTAssertEqual(config.backbone.modelType, "colqwen3")
        XCTAssertEqual(config.backbone.textConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.backbone.textConfiguration.numHiddenLayers, 1)
        XCTAssertEqual(config.backbone.visionConfiguration.outHiddenSize, 16)
        XCTAssertEqual(config.backbone.imageTokenIndex, 9)
        XCTAssertEqual(config.backbone.visionStartTokenId, 11)
    }

    func testMissingEmbedDimFailsToDecode() {
        let json = SharpAIColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "")

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                SharpAIColqwen3Configuration.self, from: Data(json.utf8)))
    }

    func testZeroEmbedDimFailsValidation() throws {
        let json = SharpAIColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "\"embed_dim\": 0,")
        let config = try JSONDecoder().decode(
            SharpAIColqwen3Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTinyConfigPassesValidation() throws {
        let config = try SharpAIColqwen3TestFixtures.tinyConfiguration()
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
