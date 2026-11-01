// SharpAI fork overlay: InternVL-style dynamic tiling for eagle_2_5_vl. Not part of upstream.
//
// Port of vLLM's transformers_utils/processors/internvl.py (resolve_internvl_min_max_num,
// get_internvl_target_ratios, calculate_internvl_targets, dynamic_preprocess_internvl).
// UNVERIFIED: Python sorts a set of ratios by area and keeps set order for ties. Here ties sort
// by columns, so the chosen tile grid can differ from HF only when two grids tie exactly.

import Foundation

public struct SharpAIEagle25VlTilePlan: Equatable, Sendable {
    /// Tiles across the width (`ratio[0]`).
    public let columns: Int
    /// Tiles down the height (`ratio[1]`).
    public let rows: Int
    /// True when a whole-image thumbnail tile is appended.
    public let hasThumbnail: Bool
    public let imageSize: Int

    public var blocks: Int { columns * rows }
    public var tileCount: Int { blocks + (hasThumbnail ? 1 : 0) }
    public var targetWidth: Int { columns * imageSize }
    public var targetHeight: Int { rows * imageSize }

    /// Patch-count range after `resolve_internvl_min_max_num`. A thumbnail adds one to the max.
    public static func resolvedPatchRange(
        minDynamicPatch: Int, maxDynamicPatch: Int, dynamicImageSize: Bool, useThumbnail: Bool
    ) -> (min: Int, max: Int) {
        let minNum = dynamicImageSize ? minDynamicPatch : 1
        var maxNum = dynamicImageSize ? maxDynamicPatch : 1
        if useThumbnail && maxNum != 1 {
            maxNum += 1
        }
        return (minNum, maxNum)
    }

    /// Every `(columns, rows)` with `min <= columns * rows <= max`, sorted by area then columns.
    public static func targetRatios(min minNum: Int, max maxNum: Int) -> [(Int, Int)] {
        guard minNum >= 1, maxNum >= minNum else { return [] }

        var ratios: [(Int, Int)] = []
        for n in minNum ... maxNum {
            for i in 1 ... n {
                for j in 1 ... n where minNum <= i * j && i * j <= maxNum {
                    if !ratios.contains(where: { $0.0 == i && $0.1 == j }) {
                        ratios.append((i, j))
                    }
                }
            }
        }
        return ratios.sorted { lhs, rhs in
            (lhs.0 * lhs.1, lhs.0) < (rhs.0 * rhs.1, rhs.0)
        }
    }

    /// Picks the tile grid closest to the image aspect ratio, as `find_closest_aspect_ratio` does.
    public static func plan(
        width: Int, height: Int, imageSize: Int,
        minDynamicPatch: Int, maxDynamicPatch: Int,
        dynamicImageSize: Bool, useThumbnail: Bool
    ) -> SharpAIEagle25VlTilePlan {
        let range = resolvedPatchRange(
            minDynamicPatch: minDynamicPatch, maxDynamicPatch: maxDynamicPatch,
            dynamicImageSize: dynamicImageSize, useThumbnail: useThumbnail)
        let ratios = targetRatios(min: range.min, max: range.max)

        let aspect = Double(width) / Double(height)
        let area = Double(width * height)
        let tileArea = 0.5 * Double(imageSize * imageSize)

        var best = (1, 1)
        var bestDiff = Double.infinity
        for ratio in ratios {
            let diff = abs(aspect - Double(ratio.0) / Double(ratio.1))
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff, area > tileArea * Double(ratio.0 * ratio.1) {
                best = ratio
            }
        }

        let blocks = best.0 * best.1
        return SharpAIEagle25VlTilePlan(
            columns: best.0, rows: best.1,
            hasThumbnail: useThumbnail && blocks != 1,
            imageSize: imageSize)
    }
}
