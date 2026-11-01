// SharpAI fork overlay tests for the alpamayo2_super trajectory-token mask. Not part of upstream.

import Foundation
import MLX
import MLXVLM
import XCTest

final class SharpAIAlpamayo2SuperTrajectoryMaskTests: XCTestCase {

    func testAdditiveMaskMarksOnlyTrajectoryTokens() throws {
        // Direction verified against NVlabs/alpamayo2 alpamayo2_super.py (-inf on the trajectory span during CoT generation).
        let mask = try SharpAIAlpamayo2SuperTrajectoryMask.additiveMask(
            vocabularySize: 8, trajectoryTokens: 5 ..< 8)
        let values = mask.asArray(Float.self)

        XCTAssertEqual(mask.shape, [8])
        XCTAssertEqual(Array(values[0 ..< 5]), [0, 0, 0, 0, 0])
        for id in 5 ..< 8 {
            XCTAssertTrue(values[id].isInfinite && values[id] < 0, "id \(id)")
        }
    }

    func testApplyMasksLastAxisAndKeepsShape() throws {
        let logits = MLXArray(Array(repeating: Float(1), count: 16), [2, 8])

        let result = try SharpAIAlpamayo2SuperTrajectoryMask.apply(
            logits, trajectoryTokens: 6 ..< 8)

        XCTAssertEqual(result.shape, [2, 8])
        let values = result.asArray(Float.self)
        XCTAssertEqual(values[1], 1)
        XCTAssertTrue(values[6].isInfinite && values[6] < 0)
        XCTAssertEqual(values[8 + 1], 1)
        XCTAssertTrue(values[8 + 7].isInfinite && values[8 + 7] < 0)
    }

    func testRangeOutsideVocabThrows() {
        XCTAssertThrowsError(
            try SharpAIAlpamayo2SuperTrajectoryMask.additiveMask(
                vocabularySize: 8, trajectoryTokens: 6 ..< 9))
        XCTAssertThrowsError(
            try SharpAIAlpamayo2SuperTrajectoryMask.additiveMask(
                vocabularySize: 8, trajectoryTokens: 3 ..< 3))
    }
}
