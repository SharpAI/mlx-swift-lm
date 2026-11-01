// SharpAI fork overlay tests for domynedge configuration. Not part of upstream.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIDomynedgeConfigurationTests: XCTestCase {

    func testDecodesTinyConfiguration() throws {
        let config = try SharpAIDomynedgeTestFixtures.tinyConfiguration()

        XCTAssertEqual(config.modelType, "domynedge")
        XCTAssertEqual(config.hiddenSize, 16)
        XCTAssertEqual(config.hiddenLayers, 4)
        XCTAssertEqual(config.attentionHeads, 4)
        XCTAssertEqual(config.kvHeads, 2)
        XCTAssertEqual(config.vocabularySize, 64)
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertNoThrow(try config.validateModelConfiguration())
    }

    func testLayerSpecsResolvePerLayerHeadSizeRopeAndWindow() throws {
        let specs = try SharpAIDomynedgeTestFixtures.tinyConfiguration().layerSpecs()

        XCTAssertEqual(specs.count, 4)
        XCTAssertEqual(
            specs[0], SharpAIDomynedgeLayerSpec(headDim: 16, usesRope: false, slidingWindow: nil))
        XCTAssertEqual(
            specs[1], SharpAIDomynedgeLayerSpec(headDim: 8, usesRope: true, slidingWindow: 4))
        XCTAssertEqual(
            specs[2], SharpAIDomynedgeLayerSpec(headDim: 8, usesRope: true, slidingWindow: 4))
        XCTAssertEqual(
            specs[3], SharpAIDomynedgeLayerSpec(headDim: 16, usesRope: false, slidingWindow: nil))
    }

    func testDefaultsWhenOptionalKeysAreAbsent() throws {
        let json = """
            {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "vocab_size": 64
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: Data(json.utf8))

        XCTAssertEqual(config.modelType, "domynedge")
        XCTAssertEqual(config.kvHeads, 4)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.hiddenAct, "silu")
        XCTAssertTrue(config.tieWordEmbeddings)
        XCTAssertEqual(
            config.layerSpecs(),
            [
                SharpAIDomynedgeLayerSpec(headDim: 4, usesRope: true, slidingWindow: nil),
                SharpAIDomynedgeLayerSpec(headDim: 4, usesRope: true, slidingWindow: nil),
            ])
    }

    func testValidationRejectsPositionLayersOfWrongLength() throws {
        let json = """
            {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "vocab_size": 64,
                "position_layers": ["nope"]
            }
            """
        let config = try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration()) { error in
            guard case SharpAIDomynedgeError.invalidConfiguration = error else {
                XCTFail("unexpected error: \(error)")
                return
            }
        }
    }

    func testValidationRejectsUnknownPositionKind() throws {
        let json = SharpAIDomynedgeTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"nope\", \"rope\", \"rope\", \"nope\"", with: "\"alibi\", \"rope\", \"rope\", \"nope\"")
        let config = try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    func testValidationRejectsUnsupportedActivation() throws {
        let json = SharpAIDomynedgeTestFixtures.tinyJSON.replacingOccurrences(
            of: "\"hidden_act\": \"silu\"", with: "\"hidden_act\": \"gelu\"")
        let config = try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: Data(json.utf8))

        XCTAssertThrowsError(try config.validateModelConfiguration())
    }

    /// Reads the real fixture from the repo, so the layout checked here is the checkpoint's.
    func testRealFixtureLayerLayout() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Domynedge
            .deletingLastPathComponent()  // SharpAI
            .deletingLastPathComponent()  // MLXLMTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent(
                "scripts/conformance/offline/fixtures/domynedge/domyn__Domyn-Edge-Base.json")
        let data = try Data(contentsOf: url)

        let config = try JSONDecoder.json5().decode(
            SharpAIDomynedgeConfiguration.self, from: data)
        try config.validateModelConfiguration()
        let specs = config.layerSpecs()

        XCTAssertEqual(specs.count, 36)
        XCTAssertEqual(config.attentionHeads, 32)
        XCTAssertEqual(config.kvHeads, 8)

        // Full-attention layers are exactly the nope layers in the fixture.
        let fullLayers = specs.indices.filter { specs[$0].slidingWindow == nil }
        XCTAssertEqual(fullLayers, [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 34, 35])
        XCTAssertTrue(specs.allSatisfy { $0.usesRope == ($0.slidingWindow != nil) })
        XCTAssertEqual(specs[0].headDim, 256)
        XCTAssertEqual(specs[2].headDim, 64)
        XCTAssertEqual(specs[2].slidingWindow, 1024)
    }
}
