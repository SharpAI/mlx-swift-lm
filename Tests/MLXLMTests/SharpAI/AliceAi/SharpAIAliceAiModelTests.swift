// SharpAI fork overlay tests for alice_ai forward pass, cache use, and registration. Not part of upstream.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIAliceAiModelTests: XCTestCase {

    private func makeModel() throws -> SharpAIAliceAiModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIAliceAiConfiguration.self, from: try SharpAIAliceAiTestSupport.tinyData())
        try config.validateModelConfiguration()
        return SharpAIAliceAiModel(config)
    }

    func testForwardReturnsVocabularyLogits() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(3), 7, 11]).reshaped(1, 3)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 3, 32])
        XCTAssertTrue(logits.asType(.float32).asArray(Float.self).allSatisfy { $0.isFinite })
    }

    /// Prefill two tokens into the caches, then decode the third. Must match the full pass at that position.
    func testCachedDecodeMatchesFullForward() throws {
        let model = try makeModel()
        let tokens = MLXArray([Int32(3), 7, 11]).reshaped(1, 3)
        let vocabulary = 32

        let full = model(tokens, cache: nil).asType(.float32).asArray(Float.self)
        let cache = model.makeCache()
        _ = model(tokens[0..., 0 ..< 2], cache: cache)
        let last = model(tokens[0..., 2 ..< 3], cache: cache).asType(.float32).asArray(Float.self)

        SharpAIAliceAiTestSupport.assertClose(
            last, Array(full[(2 * vocabulary)...]), tolerance: 1e-3)
    }

    func testCreatorRegistersAliceAiKey() throws {
        let creator = try XCTUnwrap(SharpAIAliceAiOverlay.creators["alice_ai"])
        let model = try creator(try SharpAIAliceAiTestSupport.tinyData())
        XCTAssertTrue(model is SharpAIAliceAiModel)
    }

    func testCreatorRejectsInvalidConfiguration() throws {
        let creator = try XCTUnwrap(SharpAIAliceAiOverlay.creators["alice_ai"])
        let data = try SharpAIAliceAiTestSupport.tinyData(
            overrides: ["layer_types": ["linear_attention"]])
        XCTAssertThrowsError(try creator(data))
    }
}
