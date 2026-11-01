// SharpAI fork overlay tests for the ModernVBERT model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random init weights, no network.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIModernvbertModelTests: XCTestCase {

    /// Text width 16, vision width 8, 32x32 image, 8-pixel patches, 2x2 shuffle (4 image tokens).
    private func tinyConfiguration(architectures: String = "\"ColModernVBertForRetrieval\"") throws
        -> SharpAIModernvbertConfiguration
    {
        let json = """
            {
              "architectures": [\(architectures)],
              "model_type": "modernvbert",
              "embedding_dim": 16,
              "image_token_id": 60,
              "pixel_shuffle_factor": 2,
              "text_config": {
                "vocab_size": 64,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 3,
                "num_attention_heads": 2,
                "global_attn_every_n_layers": 2,
                "local_attention": 4
              },
              "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "patch_size": 8,
                "image_size": 32
              }
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIModernvbertConfiguration.self, from: Data(json.utf8))
    }

    private func assertUnitNorm(
        _ embeddings: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        let squaredNorms = (embeddings * embeddings).sum(axis: -1).asArray(Float.self)
        XCTAssertFalse(squaredNorms.isEmpty, file: file, line: line)
        for value in squaredNorms {
            XCTAssertEqual(value, 1, accuracy: 1e-3, file: file, line: line)
        }
    }

    func testTextOnlyEmbeddingHasExpectedShapeAndUnitNorm() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let embeddings = try model.embed(ids)

        XCTAssertEqual(embeddings.shape, [1, 5, 16])
        assertUnitNorm(embeddings)
    }

    func testImageTokensAreMergedIntoSequence() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        // Four image tokens, matching `imageTokenCount`.
        let ids = MLXArray([1, 2, 60, 60, 60, 60] as [Int32]).reshaped(1, 6)
        let pixels = MLXArray.zeros([1, 3, 32, 32])

        let embeddings = try model.embed(ids, pixelValues: pixels)

        XCTAssertEqual(embeddings.shape, [1, 6, 16])
        assertUnitNorm(embeddings)
    }

    func testImageTokenCountMismatchThrows() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        // Only three image tokens for a 32x32 image that needs four.
        let ids = MLXArray([1, 2, 60, 60, 60] as [Int32]).reshaped(1, 5)
        let pixels = MLXArray.zeros([1, 3, 32, 32])

        XCTAssertThrowsError(try model.embed(ids, pixelValues: pixels)) { error in
            XCTAssertEqual(
                error as? SharpAIModernvbertError,
                .imageTokenMismatch(expected: 4, actual: 3))
        }
    }

    func testPaddedTokensAreZeroed() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3, 0, 0] as [Int32]).reshaped(1, 5)
        let mask = MLXArray([1, 1, 1, 0, 0] as [Int32]).reshaped(1, 5)

        let embeddings = try model.embed(ids, attentionMask: mask)

        XCTAssertEqual(embeddings.shape, [1, 5, 16])
        XCTAssertEqual(embeddings[0, 4].abs().sum().item(Float.self), 0, accuracy: 1e-6)
    }

    func testAdditiveMaskStaysFiniteInFloat16() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        let allowed = MLXArray([false, false, false, false] as [Bool]).reshaped(1, 1, 2, 2)

        let mask = model.model.textModel.additiveMask(allowed).asType(.float16)
        let weights = softmax(mask, axis: -1).asType(.float32)

        XCTAssertFalse(any(isNaN(weights)).item(Bool.self))
        XCTAssertEqual(weights[0, 0, 0, 0].item(Float.self), 0.5, accuracy: 1e-3)
    }

    func testFirstLayerHasNoAttentionNorm() throws {
        let model = SharpAIModernvbert(try tinyConfiguration())
        let layers = model.model.textModel.layers

        XCTAssertNil(layers[0].attnNorm)
        XCTAssertNotNil(layers[1].attnNorm)
        XCTAssertTrue(layers[0].isGlobal)
        XCTAssertFalse(layers[1].isGlobal)
    }

    func testPlainModelReturnsHiddenStatesWithoutProjection() throws {
        let model = SharpAIModernvbert(
            try tinyConfiguration(architectures: "\"ModernVBertModel\""))
        let ids = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        XCTAssertNil(model.embeddingProjLayer)
        let hidden = try model.embed(ids)

        XCTAssertEqual(hidden.shape, [1, 5, 16])
    }

    func testPixelShuffleKeepsElementOrderForSingleCell() {
        // A 2x2 grid with factor 2 collapses to one token. The order is row-major patches.
        let patches = MLXArray(0 ..< 12).asType(.float32).reshaped(1, 4, 3)

        let shuffled = SharpAIModernvbertVision.Connector.pixelShuffle(patches, factor: 2)

        XCTAssertEqual(shuffled.shape, [1, 1, 12])
        XCTAssertEqual(shuffled.reshaped(-1).asArray(Float.self), (0 ..< 12).map { Float($0) })
    }
}
