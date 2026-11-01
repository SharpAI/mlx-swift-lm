// SharpAI fork overlay: moondream3 crop tiling and prompt routing. Not part of upstream.
//
// Port of vLLM's transformers_utils/processors/moondream3.py: select_tiling and the task
// routing in its default chat template.
// UNVERIFIED: vLLM uses PIL LANCZOS for resizing. This port uses CoreImage bicubic, so pixel
// values differ slightly from the reference.

import Foundation

public enum SharpAIMoondream3Tiling {

    /// Grid of local crops `(rows, cols)` for an image, as `select_tiling` in vLLM.
    public static func selectTiling(
        height: Int, width: Int, cropSize: Int, maxCrops: Int
    ) -> (rows: Int, cols: Int) {
        if height <= cropSize || width <= cropSize {
            return (1, 1)
        }

        let minRows = Int((Double(height) / Double(cropSize)).rounded(.up))
        let minCols = Int((Double(width) / Double(cropSize)).rounded(.up))

        if minRows * minCols > maxCrops {
            let ratio = (Double(maxCrops) / Double(minRows * minCols)).squareRoot()
            return (
                max(1, Int((Double(minRows) * ratio).rounded(.down))),
                max(1, Int((Double(minCols) * ratio).rounded(.down)))
            )
        }

        var rows = Int((Double(maxCrops) * Double(height) / Double(width)).squareRoot().rounded(.down))
        var cols = Int((Double(maxCrops) * Double(width) / Double(height)).squareRoot().rounded(.down))
        rows = max(rows, minRows)
        cols = max(cols, minCols)

        if rows * cols > maxCrops {
            if cols > rows {
                cols = maxCrops / rows
            } else {
                rows = maxCrops / cols
            }
        }
        return (max(1, rows), max(1, cols))
    }
}

public enum SharpAIMoondream3Prompt {

    /// Text that follows the image span, as vLLM's default template renders it.
    ///
    /// `caption` and `describe` use the describe task. Their optional word after the keyword
    /// sets the length. Any other text is a query.
    public static func formatTask(_ text: String) -> String {
        if text == "caption" {
            return "<|md_reserved_0|>describe<|md_reserved_1|>normal<|md_reserved_2|>"
        }
        if text.hasPrefix("caption ") {
            return "<|md_reserved_0|>describe<|md_reserved_1|>"
                + String(text.dropFirst(8)) + "<|md_reserved_2|>"
        }
        if text == "describe" {
            return "<|md_reserved_0|>describe<|md_reserved_1|>normal<|md_reserved_2|>"
        }
        if text.hasPrefix("describe ") {
            return "<|md_reserved_0|>describe<|md_reserved_1|>"
                + String(text.dropFirst(9)) + "<|md_reserved_2|>"
        }
        return "<|md_reserved_0|>query<|md_reserved_1|>" + text + "<|md_reserved_2|>"
    }
}
