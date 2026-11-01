// SharpAI fork overlay tests for muse_glimmer_text model structure, forward pass and registration.
// Not part of upstream. Random weights only; no checkpoint is read.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class SharpAIMuseGlimmerTextModelTests: XCTestCase {

    func testTinyForwardProducesVocabLogits() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()
        let model = SharpAIMuseGlimmerTextModel(config)

        let tokens = MLXArray([1, 2, 3] as [Int32]).expandedDimensions(axis: 0)
        let cache = try model.newCache(parameters: nil)
        let logits = model(tokens, cache: cache)

        XCTAssertEqual(logits.shape, [1, 3, config.vocabularySize])
    }

    func testNewCacheHasOneEntryPerLayer() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()
        let model = SharpAIMuseGlimmerTextModel(config)

        let cache = try model.newCache(parameters: nil)

        XCTAssertEqual(cache.count, config.hiddenLayers)
    }

    func testLoRALayersMatchDecoderLayers() throws {
        let config = try SharpAIMuseGlimmerTextTestFixtures.configuration()
        let model = SharpAIMuseGlimmerTextModel(config)

        XCTAssertEqual(model.loraLayers.count, config.hiddenLayers)
    }

    func testOverlayCreatorBuildsModelFromInlineConfig() throws {
        let creator = try XCTUnwrap(SharpAIMuseGlimmerTextOverlay.creators["muse_glimmer_text"])

        let model = try creator(Data(SharpAIMuseGlimmerTextTestFixtures.tinyJSON.utf8))

        XCTAssertTrue(model is SharpAIMuseGlimmerTextModel)
    }

    func testOverlayCreatorRejectsInvalidLayerTypes() throws {
        let creator = try XCTUnwrap(SharpAIMuseGlimmerTextOverlay.creators["muse_glimmer_text"])
        let json = """
            {
                "model_type": "muse_glimmer_text",
                "num_hidden_layers": 3,
                "layer_types": ["full_attention"]
            }
            """

        XCTAssertThrowsError(try creator(Data(json.utf8)))
    }
}
