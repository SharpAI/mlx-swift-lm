import Foundation
import MLX
import MLXLMCommon
import XCTest

@testable import MLXVLM

/// Tiny random-weight forward passes. Shapes and norms only; no real weights.
final class SharpAIOpsColqwen3ForwardTests: XCTestCase {

    private func tinyModel() throws -> SharpAIOpsColqwen3Model {
        SharpAIOpsColqwen3Model(try SharpAIOpsColqwen3TestFixtures.tinyConfiguration())
    }

    func testTextForwardShapeAndUnitNorm() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped([1, 4])

        let out = try model.embed(inputIds: ids)
        eval(out)

        XCTAssertEqual(out.shape, [1, 4, 8])
        let norms = sqrt((out * out).sum(axis: -1)).asArray(Float.self)
        for norm in norms {
            XCTAssertEqual(norm, 1, accuracy: 1e-3)
        }
    }

    func testPaddedPositionIsZeroAndOutputFinite() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(0), 4, 5, 1, 2, 3]).reshaped([2, 3])
        let mask = MLXArray([Int32(1), 1, 1, 0, 1, 1]).reshaped([2, 3])

        let out = try model.embed(inputIds: ids, attentionMask: mask)
        eval(out)

        XCTAssertEqual(out.shape, [2, 3, 8])
        let values = out.asArray(Float.self)
        // Row 1, position 0 is padding; its 8 values start at (1 * 3 + 0) * 8.
        XCTAssertEqual(Array(values[24 ..< 32]), Array(repeating: Float(0), count: 8))
        XCTAssertTrue(values.allSatisfy { $0.isFinite })
    }

    /// UNVERIFIED: the image path is checked only for shapes. The grid and token layout
    /// follow Qwen3VL conventions; no real ops_colqwen3 image has been run.
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

        XCTAssertEqual(out.shape, [1, 8, 8])
        XCTAssertTrue(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testPixelValuesWithoutGridThrows() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 11, 9, 9, 9, 9, 12, 2]).reshaped([1, 8])
        let pixels = MLXRandom.normal([16, 12])

        XCTAssertThrowsError(try model.embed(inputIds: ids, pixelValues: pixels))
    }
}
