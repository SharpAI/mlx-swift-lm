import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIQwen3VlNemotronEmbedConfigTests: XCTestCase {

    func testDecodesBackboneEmbedDimAndPooling() throws {
        let config = try SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfiguration()

        XCTAssertEqual(config.embedDim, 8)
        XCTAssertEqual(config.pooling, .last)
        XCTAssertEqual(config.backbone.modelType, "qwen3_vl_nemotron_embed")
        XCTAssertEqual(config.backbone.textConfiguration.hiddenSize, 16)
        XCTAssertEqual(config.backbone.visionConfiguration.outHiddenSize, 16)
        XCTAssertEqual(config.backbone.imageTokenIndex, 9)
        XCTAssertEqual(config.backbone.visionStartTokenId, 11)
    }

    func testEmbedDimDefaultsToTextHiddenSize() throws {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.embedDim, 16)
    }

    /// The real checkpoint config.json carries `"pooling": "colbert"` (per-token output).
    /// Verified against: https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2/resolve/main/config.json
    func testRealCheckpointColbertPoolingDecodes() throws {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"pooling\": \"last\"", with: "\"pooling\": \"colbert\"")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.pooling, .colbert)
    }

    func testPoolingDefaultsToColbertWhenAbsent() throws {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"pooling\": \"last\",", with: "")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.pooling, .colbert)
    }

    func testMeanPoolingDecodes() throws {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"pooling\": \"last\"", with: "\"pooling\": \"mean\"")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.pooling, .mean)
    }

    /// UNVERIFIED: the set of pooling names is a local choice, not from a checkpoint.
    func testUnknownPoolingFailsToDecode() {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"pooling\": \"last\"", with: "\"pooling\": \"max\"")

        XCTAssertThrowsError(
            try JSONDecoder().decode(
                SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8)))
    }

    func testZeroEmbedDimFailsValidation() throws {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON
            .replacingOccurrences(of: "\"embed_dim\": 8,", with: "\"embed_dim\": 0,")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testTinyConfigPassesValidation() throws {
        let config = try SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfiguration()

        XCTAssertNoThrow(try config.validateModelConfiguration())
    }
}
