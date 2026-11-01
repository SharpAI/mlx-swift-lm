import Foundation
import XCTest

@testable import MLXVLM

final class SharpAIEagle25VlTilingTests: XCTestCase {

    private func plan(
        width: Int, height: Int, maxPatches: Int = 12, dynamic: Bool = true, thumbnail: Bool = true
    ) -> SharpAIEagle25VlTilePlan {
        SharpAIEagle25VlTilePlan.plan(
            width: width, height: height, imageSize: 448,
            minDynamicPatch: 1, maxDynamicPatch: maxPatches,
            dynamicImageSize: dynamic, useThumbnail: thumbnail)
    }

    func testSquareImageIsOneTileWithoutThumbnail() {
        let tiling = plan(width: 448, height: 448)

        XCTAssertEqual(tiling.columns, 1)
        XCTAssertEqual(tiling.rows, 1)
        XCTAssertFalse(tiling.hasThumbnail)
        XCTAssertEqual(tiling.tileCount, 1)
    }

    func testWideImageGetsTwoTilesAndThumbnail() {
        let tiling = plan(width: 896, height: 448)

        XCTAssertEqual(tiling.columns, 2)
        XCTAssertEqual(tiling.rows, 1)
        XCTAssertTrue(tiling.hasThumbnail)
        XCTAssertEqual(tiling.tileCount, 3)
        XCTAssertEqual(tiling.targetWidth, 896)
        XCTAssertEqual(tiling.targetHeight, 448)
    }

    func testTallImageGetsTwoRows() {
        let tiling = plan(width: 448, height: 896)

        XCTAssertEqual(tiling.columns, 1)
        XCTAssertEqual(tiling.rows, 2)
        XCTAssertEqual(tiling.tileCount, 3)
    }

    func testThumbnailOffKeepsOnlyGridTiles() {
        let tiling = plan(width: 896, height: 448, thumbnail: false)

        XCTAssertFalse(tiling.hasThumbnail)
        XCTAssertEqual(tiling.tileCount, 2)
    }

    func testStaticImageSizeUsesOneTile() {
        let tiling = plan(width: 896, height: 448, dynamic: false)

        XCTAssertEqual(tiling.tileCount, 1)
    }

    func testMaxOnePatchDisablesThumbnailBump() {
        // With max 1, the thumbnail bump is skipped, so only the 1x1 grid is allowed.
        let tiling = plan(width: 896, height: 448, maxPatches: 1)

        XCTAssertEqual(tiling.tileCount, 1)
    }

    func testResolvedPatchRangeAddsOneForThumbnail() {
        let withThumbnail = SharpAIEagle25VlTilePlan.resolvedPatchRange(
            minDynamicPatch: 1, maxDynamicPatch: 12, dynamicImageSize: true, useThumbnail: true)
        let withoutThumbnail = SharpAIEagle25VlTilePlan.resolvedPatchRange(
            minDynamicPatch: 1, maxDynamicPatch: 12, dynamicImageSize: true, useThumbnail: false)

        XCTAssertEqual(withThumbnail.min, 1)
        XCTAssertEqual(withThumbnail.max, 13)
        XCTAssertEqual(withoutThumbnail.max, 12)
    }

    func testTargetRatiosCoverProductRangeWithoutDuplicates() {
        let ratios = SharpAIEagle25VlTilePlan.targetRatios(min: 1, max: 4)

        // Pairs with product <= 4: (1,1), (1,2), (2,1), (1,3), (3,1), (1,4), (2,2), (4,1).
        XCTAssertEqual(ratios.count, 8)
        XCTAssertEqual(ratios.first?.0, 1)
        XCTAssertEqual(ratios.first?.1, 1)
        XCTAssertTrue(ratios.allSatisfy { $0.0 * $0.1 <= 4 })
        XCTAssertEqual(Set(ratios.map { "\($0.0)x\($0.1)" }).count, 8)
    }
}
