// SharpAI fork overlay tests for the isaac patch helpers. Not part of upstream.
// Offline only: pixel values are generated in code.

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIIsaacImageProcessorTests: XCTestCase {

    /// A 1x4x4x3 image whose value at (y, x, c) is `y * 12 + x * 3 + c`.
    private func ramp() -> MLXArray {
        MLXArray((0 ..< 48).map { Float($0) }).reshaped(1, 4, 4, 3)
    }

    func testPatchifyShapeIsGridByPatchVector() {
        let patches = SharpAIIsaacImageProcessor.patchify(ramp(), patchSize: 2)
        XCTAssertEqual(patches.shape, [1, 4, 12])
    }

    func testPatchifyFlattensEachPatchAsRowColumnChannel() {
        let patches = SharpAIIsaacImageProcessor.patchify(ramp(), patchSize: 2)
        let values = patches.asArray(Float.self)

        // Top-left patch: rows 0-1, columns 0-1.
        XCTAssertEqual(
            Array(values[0 ..< 12]),
            [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17])
        // Top-right patch: rows 0-1, columns 2-3.
        XCTAssertEqual(
            Array(values[12 ..< 24]),
            [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23])
    }

    func testNormalizeMapsPixelRangeToMinusOneToOne() {
        let scaled = SharpAIIsaacImageProcessor.normalize(MLXArray([Float(0), 127.5, 255]))
        let values = scaled.asArray(Float.self)
        XCTAssertEqual(values[0], -1, accuracy: 1e-5)
        XCTAssertEqual(values[1], 0, accuracy: 1e-5)
        XCTAssertEqual(values[2], 1, accuracy: 1e-5)
    }
}
