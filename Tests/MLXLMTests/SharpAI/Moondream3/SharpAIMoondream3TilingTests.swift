// SharpAI fork overlay: moondream3 tiling and prompt tests. Not part of upstream.
//
// Expected grids were worked out by hand from vLLM's select_tiling with a 266 px window
// (378 px crop, 14 px patches, overlap 4).

import Foundation
import MLX
import XCTest

@testable import MLXVLM

final class SharpAIMoondream3TilingTests: XCTestCase {

    private let window = 266

    func testSmallImageIsOneTile() {
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: 266, width: 900, cropSize: window, maxCrops: 12)

        XCTAssertEqual(tiling.rows, 1)
        XCTAssertEqual(tiling.cols, 1)
    }

    func testSquareImageUsesSquareGrid() {
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: 378, width: 378, cropSize: window, maxCrops: 12)

        XCTAssertEqual(tiling.rows, 3)
        XCTAssertEqual(tiling.cols, 3)
    }

    func testWideImageFollowsAspectRatio() {
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: 300, width: 600, cropSize: window, maxCrops: 12)

        XCTAssertEqual(tiling.rows, 2)
        XCTAssertEqual(tiling.cols, 4)
    }

    func testLargeImageIsCappedAtMaxCrops() {
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: 1000, width: 2000, cropSize: window, maxCrops: 12)

        XCTAssertEqual(tiling.rows, 2)
        XCTAssertEqual(tiling.cols, 4)
        XCTAssertLessThanOrEqual(tiling.rows * tiling.cols, 12)
    }

    func testExtremeAspectRatioUnderTightCap() {
        let tiling = SharpAIMoondream3Tiling.selectTiling(
            height: 300, width: 2000, cropSize: window, maxCrops: 4)

        XCTAssertEqual(tiling.rows, 1)
        XCTAssertEqual(tiling.cols, 4)
    }

    func testTaskRoutingForCaptionAndDescribe() {
        XCTAssertEqual(
            SharpAIMoondream3Prompt.formatTask("caption"),
            "<|md_reserved_0|>describe<|md_reserved_1|>normal<|md_reserved_2|>")
        XCTAssertEqual(
            SharpAIMoondream3Prompt.formatTask("caption short"),
            "<|md_reserved_0|>describe<|md_reserved_1|>short<|md_reserved_2|>")
        XCTAssertEqual(
            SharpAIMoondream3Prompt.formatTask("describe long"),
            "<|md_reserved_0|>describe<|md_reserved_1|>long<|md_reserved_2|>")
    }

    func testOtherTextIsAQuery() {
        XCTAssertEqual(
            SharpAIMoondream3Prompt.formatTask("What is this?"),
            "<|md_reserved_0|>query<|md_reserved_1|>What is this?<|md_reserved_2|>")
    }

    func testReconstructStitchesOverlapFreeCropsSideBySide() {
        // Two 4x4 crops, overlap 1: output is 4 rows by 6 columns.
        let crops = MLX.concatenated(
            [MLXArray.ones([1, 4, 4, 1]), MLXArray.ones([1, 4, 4, 1]) * 2], axis: 0)

        let grid = SharpAIMoondream3Vision.reconstruct(
            crops, tiling: (rows: 1, cols: 2), overlapMargin: 1)

        XCTAssertEqual(grid.shape, [4, 6, 1])
        let firstRow = grid[0, 0..., 0].asArray(Float.self)
        XCTAssertEqual(firstRow, [1, 1, 1, 2, 2, 2])
    }

    func testAdaptivePoolAveragesBins() {
        let grid = MLXArray.ones([4, 4, 2])

        let pooled = SharpAIMoondream3Vision.adaptiveAveragePool(grid, outputSize: 2)

        XCTAssertEqual(pooled.shape, [2, 2, 2])
        XCTAssertTrue(pooled.asArray(Float.self).allSatisfy { abs($0 - 1) < 1e-5 })
    }

    func testPatchifyKeepsChannelMajorOrder() {
        let pixels = MLXArray((0 ..< 8).map { Float($0) }).reshaped(1, 2, 2, 2)

        let patches = SharpAIMoondream3Vision.patchify(pixels, patchSize: 2)

        XCTAssertEqual(patches.shape, [1, 1, 8])
        XCTAssertEqual(patches.asArray(Float.self), (0 ..< 8).map { Float($0) })
    }
}
