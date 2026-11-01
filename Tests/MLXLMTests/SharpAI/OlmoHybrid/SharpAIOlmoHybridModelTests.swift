// SharpAI fork overlay tests for olmo_hybrid forward pass, cache use, and registration. Not part of upstream.
//
// UNVERIFIED: the GatedDeltaNet recurrence runs in SharpAIOverlayGatedDeltaKDA. Its Metal kernel path
// is not covered here, because these shapes (Dk = 4) take the ops fallback. Batched left padding is not
// tested.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import XCTest

final class SharpAIOlmoHybridModelTests: XCTestCase {

    private func makeModel(overrides: [String: Any] = [:]) throws -> SharpAIOlmoHybridModel {
        let config = try JSONDecoder.json5().decode(
            SharpAIOlmoHybridConfiguration.self,
            from: try SharpAIOlmoHybridTestSupport.tinyData(overrides: overrides))
        try config.validateModelConfiguration()
        return SharpAIOlmoHybridModel(config)
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

        SharpAIOlmoHybridTestSupport.assertClose(
            last, Array(full[(2 * vocabulary)...]), tolerance: 1e-3)
    }

    /// Without rope_parameters the full-attention layers run NoPE. The forward pass must still work.
    func testForwardWithoutRopeParameters() throws {
        let model = try makeModel(overrides: ["rope_parameters": NSNull()])
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 4, 32])
    }

    func testTiedEmbeddingsUseEmbeddingAsLmHead() throws {
        let model = try makeModel(overrides: ["tie_word_embeddings": true])
        let tokens = MLXArray([Int32(5)]).reshaped(1, 1)

        let logits = model(tokens, cache: nil)

        XCTAssertEqual(logits.shape, [1, 1, 32])
    }

    func testLoraLayersAreTheDecoderLayers() throws {
        let model = try makeModel()
        XCTAssertEqual(model.loraLayers.count, 4)
    }

    func testCreatorRegistersOlmoHybridKey() throws {
        let creator = try XCTUnwrap(SharpAIOlmoHybridOverlay.creators["olmo_hybrid"])
        let model = try creator(try SharpAIOlmoHybridTestSupport.tinyData())
        XCTAssertTrue(model is SharpAIOlmoHybridModel)
    }

    func testCreatorRejectsInvalidConfiguration() throws {
        let creator = try XCTUnwrap(SharpAIOlmoHybridOverlay.creators["olmo_hybrid"])
        let data = try SharpAIOlmoHybridTestSupport.tinyData(overrides: [
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "linear_attention"]
        ])
        XCTAssertThrowsError(try creator(data))
    }
}
