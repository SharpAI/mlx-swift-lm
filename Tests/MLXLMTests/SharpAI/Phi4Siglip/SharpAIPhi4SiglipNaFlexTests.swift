import Foundation
import MLX
import XCTest

@testable import MLXVLM

/// Pure preprocessing checks. No model weights are needed.
final class SharpAIPhi4SiglipNaFlexTests: XCTestCase {

    func testNaFlexSizeStaysWithinBudgetAndAlignsToPatches() {
        let size = SharpAIPhi4SiglipProcessor.naflexSize(
            width: 640, height: 480, patchSize: 16, maxPatches: 3600)

        XCTAssertEqual(size.height % 16, 0)
        XCTAssertEqual(size.width % 16, 0)
        let count = (size.height / 16) * (size.width / 16)
        XCTAssertLessThanOrEqual(count, 3600)
        XCTAssertGreaterThan(count, 0)

        let aspect = Double(size.width) / Double(size.height)
        XCTAssertEqual(aspect, 4.0 / 3.0, accuracy: 0.05)
    }

    func testNaFlexSizeFitsLargeImageBelowBudget() {
        let size = SharpAIPhi4SiglipProcessor.naflexSize(
            width: 8000, height: 6000, patchSize: 16, maxPatches: 3600)

        let count = (size.height / 16) * (size.width / 16)
        XCTAssertLessThanOrEqual(count, 3600)
        XCTAssertGreaterThan(count, 3000)
    }

    func testPatchifyOrdersRowsByRowColumnChannel() {
        // Channel c, row r, column col holds c * 4 + r * 2 + col.
        let values = (0 ..< 12).map { Float($0) }
        let pixels = MLXArray(values, [3, 2, 2])

        let patches = SharpAIPhi4SiglipProcessor.patchify(pixels, patchSize: 2)

        XCTAssertEqual(patches.shape, [1, 12])
        var expected: [Float] = []
        for row in 0 ..< 2 {
            for col in 0 ..< 2 {
                for channel in 0 ..< 3 {
                    expected.append(Float(channel * 4 + row * 2 + col))
                }
            }
        }
        XCTAssertEqual(patches.reshaped(-1).asArray(Float.self), expected)
    }

    func testPatchifyGridShape() {
        let pixels = MLXArray.zeros([3, 6, 4])

        let patches = SharpAIPhi4SiglipProcessor.patchify(pixels, patchSize: 2)

        XCTAssertEqual(patches.shape, [6, 12])
    }

    func testExpandPlaceholdersReplacesEachRunInOrder() throws {
        let tokens = [5, 7, 6, 7, 9]

        let expanded = try SharpAIPhi4SiglipProcessor.expandPlaceholders(
            tokens, placeholder: [7], imageToken: 31, counts: [2, 1])

        XCTAssertEqual(expanded, [5, 31, 31, 6, 31, 9])
    }

    func testExpandPlaceholdersThrowsOnCountMismatch() {
        XCTAssertThrowsError(
            try SharpAIPhi4SiglipProcessor.expandPlaceholders(
                [5, 7, 9], placeholder: [7], imageToken: 31, counts: [1, 1]))
    }
}
