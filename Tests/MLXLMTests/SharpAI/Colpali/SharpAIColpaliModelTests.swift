// SharpAI fork overlay tests for the ColPali model. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAIColpaliModelTests: XCTestCase {

    /// Text width 32, embedding 16, 28x28 image with 14-pixel patches (4 image tokens).
    private func tinyConfiguration() throws -> SharpAIColpaliConfiguration {
        let json = """
            {
              "model_type": "colpali",
              "embedding_dim": 16,
              "text_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 64
              },
              "vlm_config": {
                "model_type": "paligemma",
                "image_token_index": 60,
                "pad_token_id": 0,
                "vision_config": {
                  "hidden_size": 16,
                  "num_hidden_layers": 1,
                  "intermediate_size": 32,
                  "num_attention_heads": 2,
                  "patch_size": 14,
                  "image_size": 28,
                  "projection_dim": 32
                }
              }
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAIColpaliConfiguration.self, from: Data(json.utf8))
    }

    private func assertUnitNorm(_ embeddings: MLXArray, file: StaticString = #filePath, line: UInt = #line) {
        let squaredNorms = (embeddings * embeddings).sum(axis: -1).asArray(Float.self)
        XCTAssertFalse(squaredNorms.isEmpty, file: file, line: line)
        for value in squaredNorms {
            XCTAssertEqual(value, 1, accuracy: 1e-3, file: file, line: line)
        }
    }

    func testTextOnlyEmbeddingHasExpectedShapeAndUnitNorm() throws {
        let model = SharpAIColpali(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let embeddings = try model.embed(ids)

        XCTAssertEqual(embeddings.shape, [1, 5, 16])
        assertUnitNorm(embeddings)
    }

    func testImageEmbeddingMergesVisualTokens() throws {
        let model = SharpAIColpali(try tinyConfiguration())
        let pixels = MLXRandom.normal([1, 3, 28, 28])
        let ids = MLXArray([60, 60, 60, 60, 1, 2, 3] as [Int32]).reshaped(1, 7)

        let embeddings = try model.embed(ids, pixelValues: pixels)

        XCTAssertEqual(embeddings.shape, [1, 7, 16])
        assertUnitNorm(embeddings)
    }

    func testImageTokenCountMismatchThrows() throws {
        let model = SharpAIColpali(try tinyConfiguration())
        let pixels = MLXRandom.normal([1, 3, 28, 28])
        let ids = MLXArray([60, 60, 60, 1, 2, 3, 4] as [Int32]).reshaped(1, 7)

        XCTAssertThrowsError(try model.embed(ids, pixelValues: pixels)) { error in
            XCTAssertEqual(
                error as? SharpAIColpaliError,
                .imageTokenMismatch(expected: 4, actual: 3))
        }
    }

    func testLanguageModelEntryPointReturnsTheSameEmbeddings() throws {
        let model = SharpAIColpali(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let viaCall = model(ids, cache: nil)

        XCTAssertEqual(viaCall.shape, [1, 5, 16])
        assertUnitNorm(viaCall)
    }

    func testParameterKeysUseCheckpointNames() throws {
        let model = SharpAIColpali(try tinyConfiguration())
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("embedding_proj_layer.weight"))
        XCTAssertTrue(keys.contains("embedding_proj_layer.bias"))
        XCTAssertTrue(keys.contains("vlm.language_model.model.layers.0.self_attn.q_proj.weight"))
        XCTAssertTrue(keys.contains("vlm.multi_modal_projector.linear.weight"))
        XCTAssertTrue(
            keys.contains("vlm.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"))
    }

    func testLoRAExposesEveryDecoderBlock() throws {
        let model = SharpAIColpali(try tinyConfiguration())

        XCTAssertEqual(model.loraLayers.count, 2)
        XCTAssertEqual(model.kvHeads, [1, 1])
    }
}
