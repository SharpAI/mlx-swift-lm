import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIOpsColqwen3ConfigTests: XCTestCase {

    func testDecodesBackboneAndDims() throws {
        let config = try SharpAIOpsColqwen3TestFixtures.tinyConfiguration()

        XCTAssertEqual(config.dims, 8)
        XCTAssertEqual(config.backbone.modelType, "ops_colqwen3")
        XCTAssertEqual(config.backbone.textConfiguration.modelType, "qwen3_vl_text")
        XCTAssertEqual(config.backbone.textConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.backbone.textConfiguration.numHiddenLayers, 1)
        XCTAssertEqual(config.backbone.visionConfiguration.outHiddenSize, 16)
        XCTAssertEqual(config.backbone.imageTokenIndex, 9)
        XCTAssertEqual(config.backbone.visionStartTokenId, 11)
    }

    func testRealShapedConfigDecodes() throws {
        let config = try JSONDecoder().decode(
            SharpAIOpsColqwen3Configuration.self,
            from: Data(SharpAIOpsColqwen3TestFixtures.realShapedConfigJSON.utf8))

        XCTAssertEqual(config.dims, 2560)
        XCTAssertEqual(config.backbone.textConfiguration.hiddenSize, 2560)
        XCTAssertEqual(config.backbone.textConfiguration.numKeyValueHeads, 8)
        XCTAssertEqual(config.backbone.textConfiguration.ropeTheta, 5_000_000)
        XCTAssertEqual(config.backbone.visionConfiguration.outHiddenSize, 2560)
        XCTAssertEqual(config.backbone.visionConfiguration.deepstackVisualIndexes, [5, 11, 17])
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testMissingDimsFailsToDecode() {
        let json = SharpAIOpsColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"dims\": 8,", with: "")

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                SharpAIOpsColqwen3Configuration.self, from: Data(json.utf8)))
    }

    func testZeroDimsFailsValidation() throws {
        let json = SharpAIOpsColqwen3TestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"dims\": 8,", with: "\"dims\": 0,")
        let config = try JSONDecoder().decode(
            SharpAIOpsColqwen3Configuration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTinyConfigPassesValidation() throws {
        let config = try SharpAIOpsColqwen3TestFixtures.tinyConfiguration()
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
