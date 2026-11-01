// SharpAI fork overlay tests for the LlamaNemotronVL embedder. Not part of upstream.
//
// Offline checks only: a tiny synthetic config, random weights, no network.

import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

final class SharpAILlamaNemotronVlModelTests: XCTestCase {

    /// Text width 32, 28x28 image with 14-pixel patches. A 2x2 grid with factor 2 gives one image token.
    private func tinyConfiguration() throws -> SharpAILlamaNemotronVlConfiguration {
        let json = """
            {
              "model_type": "llama_nemotron_vl",
              "llm_config": {
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "vocab_size": 64
              },
              "vision_config": {
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "patch_size": 14,
                "image_size": 28
              },
              "downsample_ratio": 0.5,
              "ps_version": "v2",
              "img_context_token_id": 60
            }
            """
        return try JSONDecoder.json5().decode(
            SharpAILlamaNemotronVlConfiguration.self, from: Data(json.utf8))
    }

    func testPixelShuffleGroupsBlocksIntoChannels() {
        let grid = MLXArray.zeros([1, 2, 2, 4])

        let shuffled = SharpAILlamaNemotronVlProjector.pixelShuffle(grid, factor: 2, version: "v2")

        XCTAssertEqual(shuffled.shape, [1, 1, 1, 16])
    }

    func testPixelShuffleV1SkipsSecondTranspose() {
        // Non-constant ramp, so the element order shows whether the second transpose ran.
        let grid = MLXArray(0 ..< 32).asType(.float32).reshaped(1, 4, 4, 2)

        let v1 = SharpAILlamaNemotronVlProjector.pixelShuffle(grid, factor: 2, version: "v1")
        let v2 = SharpAILlamaNemotronVlProjector.pixelShuffle(grid, factor: 2, version: "v2")

        XCTAssertEqual(v1.shape, [1, 2, 2, 8])
        XCTAssertEqual(v2.shape, [1, 2, 2, 8])
        XCTAssertNotEqual(v1.asArray(Float.self), v2.asArray(Float.self))
        XCTAssertEqual(v2.asArray(Float.self), v1.transposed(0, 2, 1, 3).asArray(Float.self))
    }

    func testTextOnlyEmbedHasHiddenStateShape() throws {
        let model = SharpAILlamaNemotronVl(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5)

        let hidden = try model.embed(ids)

        XCTAssertEqual(hidden.shape, [1, 5, 32])
    }

    func testImageEmbedMergesOneVisualTokenPerImage() throws {
        let model = SharpAILlamaNemotronVl(try tinyConfiguration())
        let pixels = MLXRandom.normal([1, 3, 28, 28])
        let ids = MLXArray([60, 1, 2] as [Int32]).reshaped(1, 3)

        let hidden = try model.embed(ids, pixelValues: pixels)

        XCTAssertEqual(hidden.shape, [1, 3, 32])
    }

    func testImageTokenCountMismatchThrows() throws {
        let model = SharpAILlamaNemotronVl(try tinyConfiguration())
        let pixels = MLXRandom.normal([1, 3, 28, 28])
        let ids = MLXArray([60, 60, 1, 2] as [Int32]).reshaped(1, 4)

        XCTAssertThrowsError(try model.embed(ids, pixelValues: pixels)) { error in
            XCTAssertEqual(
                error as? SharpAILlamaNemotronVlError,
                .imageTokenMismatch(expected: 1, actual: 2))
        }
    }

    /// UNVERIFIED: the pooling is a guess. This only checks shape, not the reference output.
    func testMeanPooledCollapsesSequenceAxis() throws {
        let model = SharpAILlamaNemotronVl(try tinyConfiguration())
        let ids = MLXArray([1, 2, 3] as [Int32]).reshaped(1, 3)

        let pooled = try model.meanPooled(ids)

        XCTAssertEqual(pooled.shape, [1, 32])
    }
}
