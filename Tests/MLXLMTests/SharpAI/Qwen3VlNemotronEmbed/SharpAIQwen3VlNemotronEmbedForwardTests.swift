import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny random-weight forward passes. Shapes, norms and padding only; no real weights.
///
/// Output is per token, `[batch, seq, text hidden]` (hidden = 16 in the tiny fixture), following
/// modeling_qwen3_vl_nemotron_embed.py: no projection head, bidirectional attention, final norm skipped,
/// padded rows masked, then L2-normalized.
final class SharpAIQwen3VlNemotronEmbedForwardTests: XCTestCase {

    private func tinyModel(pooling: String = "colbert") throws -> SharpAIQwen3VlNemotronEmbedModel {
        let json = SharpAIQwen3VlNemotronEmbedTestFixtures.tinyConfigJSON.replacingOccurrences(
            of: "\"pooling\": \"last\"", with: "\"pooling\": \"\(pooling)\"")
        let config = try JSONDecoder().decode(
            SharpAIQwen3VlNemotronEmbedConfiguration.self, from: Data(json.utf8))
        return SharpAIQwen3VlNemotronEmbedModel(config)
    }

    func testTextForwardShapeAndUnitNormPerToken() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])

        let out = try model.embed(inputIds: ids)
        eval(out)

        XCTAssertEqual(out.shape, [1, 4, 16])
        let norms = sqrt((out * out).sum(axis: -1)).asArray(Float.self)
        XCTAssertEqual(norms.count, 4)
        for norm in norms {
            XCTAssertEqual(norm, 1, accuracy: 1e-3)
        }
    }

    func testPaddedBatchZeroesPadAndNormalizesRealTokens() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(0), 4, 5, 1, 2, 3]).reshaped([2, 3])
        let mask = MLXArray([Int32(1), 1, 1, 0, 1, 1]).reshaped([2, 3])

        let out = try model.embed(inputIds: ids, attentionMask: mask)
        eval(out)

        XCTAssertEqual(out.shape, [2, 3, 16])
        let values = out.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite })

        // Row (batch 1, token 0) is padding: flat offset 1 * 3 * 16 = 48, length 16. It must be all zero.
        XCTAssertEqual(Array(values[48 ..< 64]), [Float](repeating: 0, count: 16))

        // Real tokens are unit length. Flattened order is (0,0) (0,1) (0,2) (1,0) (1,1) (1,2).
        let norms = sqrt((out * out).sum(axis: -1)).asArray(Float.self)
        let expected: [Float] = [1, 1, 1, 0, 1, 1]
        for (norm, want) in zip(norms, expected) {
            XCTAssertEqual(norm, want, accuracy: 1e-3)
        }
    }

    /// UNVERIFIED: the image path is checked only for shapes. The grid and token layout
    /// follow Qwen3VL conventions; no real checkpoint has been run.
    func testImageForwardShape() throws {
        let model = try tinyModel()
        // One 4x4 patch grid merges to 4 visual tokens, bracketed by vision start and end.
        let ids = MLXArray([Int32(1), 11, 9, 9, 9, 9, 12, 2]).reshaped([1, 8])
        let pixels = MLXRandom.normal([16, 12])
        let mask = MLXArray([Int32](repeating: 1, count: 8)).reshaped([1, 8])

        let out = try model.embed(
            inputIds: ids,
            pixelValues: pixels,
            imageGridTHW: [THW(1, 4, 4)],
            attentionMask: mask)
        eval(out)

        XCTAssertEqual(out.shape, [1, 8, 16])
        XCTAssertTrue(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testPixelValuesWithoutGridThrows() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 11, 9, 9, 9, 9, 12, 2]).reshaped([1, 8])
        let pixels = MLXRandom.normal([16, 12])

        XCTAssertThrowsError(try model.embed(inputIds: ids, pixelValues: pixels))
    }
}
