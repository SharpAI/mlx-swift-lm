// SharpAI fork overlay: MiniCPM-V 4.7 canvas M-RoPE position ids. Not part of upstream mlx-swift-lm.
//
// Port of MiniCPMV4_7Model._group_visual_frames, get_vision_position_ids and get_rope_index in
// transformers modeling_minicpmv4_7.py, for one unpadded sequence (batch 1).
//
// The HF path takes `mm_token_type_ids` from the processor. Here the modality is derived from the
// placeholder ids instead: image token -> 1, video token -> 2, everything else -> 0. Markers
// (`<image>`, `</image>`, `<slice>`, `</slice>`, newline) stay text, as in HF.

import Foundation

struct SharpAIMinicpmv47CanvasPositions {
    struct Frame {
        let modality: Int
        /// First and one-past-last placeholder index of the thumbnail.
        let thumbStart: Int
        let thumbEnd: Int
        /// Detail crops that tile this frame, as placeholder index ranges.
        var slices: [(start: Int, end: Int)]
        /// One past the `</image>` / `</slice>` markers that close the frame.
        var frameEnd: Int
    }

    struct Group {
        /// Index of the `<image>` marker that opens the group.
        let start: Int
        /// One past the last marker that closes the group.
        var end: Int
        var frames: [Frame]
    }

    let imageStartId: Int
    let imageEndId: Int
    let sliceStartId: Int
    let sliceEndId: Int
    let newlineId: Int
    /// Patch-to-LLM-token factor: 4 in `"16x"` mode, 2 in `"4x"` mode.
    let mergeFactor: Int

    /// Returns `[T, H, W]` rows of length `ids.count`, and the offset that generation adds to the
    /// 1-D positions of tokens after the prompt (HF `rope_deltas`).
    /// `imageGrids` and `videoGrids` list `(h, w)` patch grids per crop, in prompt order.
    func compute(
        tokenIds ids: [Int], modalities: [Int], imageGrids: [(Int, Int)], videoGrids: [(Int, Int)]
    ) throws -> (positions: [[Int]], delta: Int) {
        let seqLen = ids.count
        guard seqLen > 0 else { return ([[], [], []], 0) }
        var rows = Array(repeating: Array(repeating: 0, count: seqLen), count: 3)
        var cursors = [1: 0, 2: 0]

        func takeGrid(_ modality: Int) throws -> (Int, Int) {
            let grids = modality == 1 ? imageGrids : videoGrids
            let index = cursors[modality] ?? 0
            guard index < grids.count else {
                throw SharpAIMinicpmv47Error.canvasGridMismatch(
                    "modality \(modality) needs crop \(index + 1), got \(grids.count)")
            }
            cursors[modality] = index + 1
            return grids[index]
        }

        func fill(_ range: Range<Int>, base: Int) {
            for index in range {
                let value = base + (index - range.lowerBound)
                for row in 0 ..< 3 {
                    rows[row][index] = value
                }
            }
        }

        func write(_ values: [[Int]], at start: Int, count expected: Int) throws {
            guard values[0].count == expected else {
                throw SharpAIMinicpmv47Error.placeholderMismatch(
                    expected: expected, actual: values[0].count)
            }
            for row in 0 ..< 3 {
                for k in 0 ..< expected {
                    rows[row][start + k] = values[row][k]
                }
            }
        }

        var currentPos = 0
        var currentIdx = 0
        for group in groupVisualFrames(ids: ids, modalities: modalities) {
            // Text in front of the group is plain 1-D.
            if group.start > currentIdx {
                fill(currentIdx ..< group.start, base: currentPos)
                currentPos += group.start - currentIdx
            }

            var frameCursor = group.start
            for frame in group.frames {
                let thumb = try takeGrid(frame.modality)
                let frameStart = frame.thumbStart - 1

                // Text between two frames is 1-D. The extra step keeps the next halo off the last of them.
                if frameStart > frameCursor {
                    let gapLen = frameStart - frameCursor
                    fill(frameCursor ..< frameStart, base: currentPos)
                    currentPos += gapLen + 1
                }

                let frameEnd = min(frame.frameEnd, group.end)
                let canvasOrigin = currentPos
                let haloBeforeCanvas = max(canvasOrigin - 1, 0)

                var canvasHeight = thumb.0 / mergeFactor
                var canvasWidth = thumb.1 / mergeFactor
                var numRows = 0
                var numCols = 0
                var llmSliceH = 0
                var llmSliceW = 0
                var firstSlice = (0, 0)
                if !frame.slices.isEmpty {
                    // Slices are laid out row-major. A gap wider than the `</slice><slice>` pair is the
                    // newline that ends a row, and it fixes the column count.
                    numCols = frame.slices.count
                    for k in 0 ..< frame.slices.count - 1
                    where frame.slices[k + 1].start - frame.slices[k].end > 2 {
                        numCols = k + 1
                        break
                    }
                    numRows = frame.slices.count / numCols
                    if numRows * numCols != frame.slices.count {
                        numRows = 1
                        numCols = frame.slices.count
                    }
                    firstSlice = try takeGrid(frame.modality)
                    llmSliceH = firstSlice.0 / mergeFactor
                    llmSliceW = firstSlice.1 / mergeFactor
                    canvasHeight = numRows * llmSliceH
                    canvasWidth = numCols * llmSliceW
                }

                // Base coat for the whole frame. `<image>` sits on the halo just outside the canvas.
                for index in frameStart ..< frameEnd {
                    for row in 0 ..< 3 {
                        rows[row][index] = canvasOrigin
                    }
                }
                rows[1][frameStart] = haloBeforeCanvas
                rows[2][frameStart] = haloBeforeCanvas

                // `</image>` closes the thumbnail on the far corner of the canvas.
                if frame.thumbEnd < frameEnd {
                    rows[1][frame.thumbEnd] = canvasOrigin + canvasHeight
                    rows[2][frame.thumbEnd] = canvasOrigin + canvasWidth
                }

                // With slices present the thumbnail is stretched over the whole canvas, so it lines up
                // with the detail crops under it. Without slices the canvas is the thumbnail grid.
                let thumbPositions = visionPositions(
                    start: canvasOrigin, grid: thumb, canvas: (canvasHeight, canvasWidth),
                    offset: (0, 0))
                try write(
                    thumbPositions, at: frame.thumbStart,
                    count: frame.thumbEnd - frame.thumbStart)

                for (k, slice) in frame.slices.enumerated() {
                    let hOffset = (k / numCols) * llmSliceH
                    let wOffset = (k % numCols) * llmSliceW
                    let target: (Int, Int)
                    if k == 0 {
                        target = firstSlice
                    } else {
                        target = try takeGrid(frame.modality)
                    }
                    let sliceH = target.0 / mergeFactor
                    let sliceW = target.1 / mergeFactor

                    // `<slice>` pins the top-left corner of its crop, `</slice>` the bottom-right.
                    let markerStart = slice.start - 1
                    if markerStart >= frameStart {
                        rows[1][markerStart] = canvasOrigin + hOffset
                        rows[2][markerStart] = canvasOrigin + wOffset
                    }
                    if slice.end < frameEnd {
                        rows[1][slice.end] = canvasOrigin + hOffset + sliceH - 1
                        rows[2][slice.end] = canvasOrigin + wOffset + sliceW - 1
                    }

                    let slicePositions = visionPositions(
                        start: canvasOrigin, grid: target, canvas: (0, 0),
                        offset: (hOffset, wOffset))
                    try write(slicePositions, at: slice.start, count: slice.end - slice.start)
                }

                // The newline that ends a row of slices sits just past the right edge of that row.
                if frame.slices.count > 1 {
                    for k in 0 ..< frame.slices.count - 1 {
                        let gapStart = frame.slices[k].end
                        let gapEnd = frame.slices[k + 1].start
                        if gapEnd - gapStart <= 2 {
                            continue
                        }
                        let boundaryH = (k / numCols + 1) * llmSliceH - 1
                        let rightEdgeW = numCols * llmSliceW
                        for newline in (gapStart + 1) ..< (gapEnd - 1) {
                            rows[1][newline] = canvasOrigin + boundaryH
                            rows[2][newline] = canvasOrigin + rightEdgeW
                        }
                    }
                }

                currentPos = canvasOrigin + max(canvasHeight, canvasWidth) + 1
                frameCursor = frameEnd
            }

            if frameCursor < group.end {
                fill(frameCursor ..< group.end, base: currentPos)
                currentPos += group.end - frameCursor
            }
            currentIdx = group.end
        }

        if currentIdx < seqLen {
            fill(currentIdx ..< seqLen, base: currentPos)
        }

        let maxPosition = rows.joined().max() ?? 0
        return (rows, maxPosition + 1 - seqLen)
    }

    /// Splits the sequence into visual groups, each a list of `thumbnail + slices` frames.
    /// Port of `_group_visual_frames`.
    func groupVisualFrames(ids: [Int], modalities: [Int]) -> [Group] {
        let markers: Set<Int> = [imageStartId, imageEndId, sliceStartId, sliceEndId, newlineId]
        let seqLen = ids.count
        var groups: [Group] = []
        // The frame that receives the next slice, as (group index, frame index).
        var openFrame: (group: Int, frame: Int)?
        var previousEnd = 0

        var index = 0
        while index < seqLen {
            let modality = modalities[index]
            var runEnd = index
            while runEnd < seqLen, modalities[runEnd] == modality {
                runEnd += 1
            }
            if modality != 0 {
                let cropStart = index
                let cropEnd = runEnd
                if let open = openFrame, cropStart >= 1, ids[cropStart - 1] == sliceStartId {
                    groups[open.group].frames[open.frame].slices.append(
                        (start: cropStart, end: cropEnd))
                } else {
                    let frame = Frame(
                        modality: modality, thumbStart: cropStart, thumbEnd: cropEnd,
                        slices: [], frameEnd: 0)
                    let gapEnd = max(previousEnd, cropStart - 1)
                    let gapIsMarkers = ids[previousEnd ..< gapEnd].allSatisfy {
                        markers.contains($0)
                    }
                    if !groups.isEmpty && gapIsMarkers {
                        groups[groups.count - 1].frames.append(frame)
                        openFrame = (groups.count - 1, groups[groups.count - 1].frames.count - 1)
                    } else {
                        groups.append(
                            Group(start: max(cropStart - 1, 0), end: 0, frames: [frame]))
                        openFrame = (groups.count - 1, 0)
                    }
                }
                previousEnd = cropEnd
            }
            index = runEnd
        }

        for groupIndex in groups.indices {
            for frameIndex in groups[groupIndex].frames.indices {
                let frame = groups[groupIndex].frames[frameIndex]
                var end = frame.slices.last?.end ?? frame.thumbEnd
                while end < seqLen, markers.contains(ids[end]) {
                    end += 1
                }
                groups[groupIndex].frames[frameIndex].frameEnd = end
            }
            groups[groupIndex].end =
                groups[groupIndex].frames.last?.frameEnd ?? groups[groupIndex].start
        }
        return groups
    }

    /// Port of `get_vision_position_ids`: `[T, H, W]` rows, one entry per LLM token of the grid.
    /// A canvas of 0 means the grid's own size, which makes the linspace an arange.
    func visionPositions(
        start: Int, grid: (Int, Int), canvas: (Int, Int), offset: (Int, Int)
    ) -> [[Int]] {
        let llmHeight = grid.0 / mergeFactor
        let llmWidth = grid.1 / mergeFactor
        let canvasHeight = canvas.0 == 0 ? llmHeight : canvas.0
        let canvasWidth = canvas.1 == 0 ? llmWidth : canvas.1

        let hCoords = Self.linspaceRounded(end: canvasHeight - 1, steps: llmHeight).map {
            $0 + offset.0
        }
        let wCoords = Self.linspaceRounded(end: canvasWidth - 1, steps: llmWidth).map {
            $0 + offset.1
        }

        var out: [[Int]] = [[], [], []]
        for h in hCoords {
            for w in wCoords {
                out[0].append(start)
                out[1].append(start + h)
                out[2].append(start + w)
            }
        }
        return out
    }

    /// `torch.linspace(0, end, steps).round()`, where round is half-to-even.
    static func linspaceRounded(end: Int, steps: Int) -> [Int] {
        if steps <= 0 {
            return []
        }
        if steps == 1 {
            return [0]
        }
        return (0 ..< steps).map { index in
            let value = Double(index) * Double(end) / Double(steps - 1)
            return Int(value.rounded(.toNearestOrEven))
        }
    }
}
